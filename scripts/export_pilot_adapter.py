"""Recover this pilot's plain, single-rank LoRA adapter without changing its source."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors.torch import load_file, save_file


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
KEY = re.compile(r"^(base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+))\.lora_([AB])\.default\.weight$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_adapter(state, adapter_config, model_config, fsdp_config):
    """Reject unsupported variants or incomplete tensors instead of guessing."""
    if fsdp_config.get("world_size") != 1 or fsdp_config.get("FSDP_version") != 1:
        raise ValueError("This exporter supports only the observed FSDP1 world-size-one checkpoint.")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected a causal-language-model LoRA adapter.")
    if model_config.get("model_type") != "qwen2":
        raise ValueError("This exporter was written for the pilot's Qwen2 checkpoint.")
    if set(adapter_config.get("target_modules", [])) != TARGETS:
        raise ValueError("Expected the seven linear projection targets from the pilot.")
    required = {"r": 8, "lora_alpha": 16, "bias": "none", "lora_dropout": 0.0}
    for name, value in required.items():
        if adapter_config.get(name) != value:
            raise ValueError(f"Unexpected {name}; expected {value!r}.")
    for name in ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias",
                 "modules_to_save", "rank_pattern", "alpha_pattern",
                 "layer_replication", "target_parameters", "trainable_token_indices",
                 "use_qalora", "alora_invocation_tokens", "use_bdlora"):
        if adapter_config.get(name):
            raise ValueError(f"Unsupported adapter option: {name}")
    if adapter_config.get("init_lora_weights") is not True:
        raise ValueError("Expected the ordinary LoRA initialization used by this pilot.")
    rank = adapter_config["r"]
    layers = model_config["num_hidden_layers"]
    recovered = {}
    seen = set()
    b_nonzero = 0
    for key, value in state.items():
        if "lora_" not in key:
            continue
        match = KEY.fullmatch(key)
        if not match:
            raise ValueError(f"Unexpected adapter key: {key}")
        prefix, layer, block, target, side = match.groups()
        if target not in TARGETS or not 0 <= int(layer) < layers:
            raise ValueError(f"Unexpected layer or target: {key}")
        expected_block = "mlp" if target in {"gate_proj", "up_proj", "down_proj"} else "self_attn"
        if block != expected_block:
            raise ValueError(f"Unexpected projection location: {key}")
        if type(value) is not torch.Tensor or value.ndim != 2:
            raise ValueError(f"Expected a full plain 2D tensor: {key}")
        base_key = prefix + ".base_layer.weight"
        base = state.get(base_key)
        if type(base) is not torch.Tensor or base.ndim != 2:
            raise ValueError(f"Missing full base projection: {base_key}")
        expected_shape = (rank, base.shape[1]) if side == "A" else (base.shape[0], rank)
        if tuple(value.shape) != expected_shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {key}")
        output_key = prefix + f".lora_{side}.weight"
        if output_key in recovered:
            raise ValueError(f"Duplicate export key: {output_key}")
        recovered[output_key] = value.detach().cpu().contiguous().clone()
        seen.add((int(layer), target, side))
        if side == "B" and torch.count_nonzero(value).item() > 0:
            b_nonzero += 1
    expected = {(layer, target, side) for layer in range(layers)
                for target in TARGETS for side in ("A", "B")}
    if seen != expected:
        raise ValueError(f"Incomplete adapter: expected {len(expected)} tensors, found {len(seen)}.")

    # Serialize only the supported architecture settings, not unused newer PEFT defaults.
    config = {name: adapter_config[name] for name in (
        "base_model_name_or_path", "task_type", "peft_type", "r",
        "lora_alpha", "lora_dropout", "target_modules", "bias",
    )}
    config.update(inference_mode=True, fan_in_fan_out=False, use_rslora=False,
                  use_dora=False, init_lora_weights=True)
    config["revision"] = adapter_config.get("revision")
    return recovered, config, b_nonzero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actor = args.actor_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists. Choose a new directory; existing exports are never overwritten.")
    if output.is_relative_to(actor):
        parser.error("Keep the recovered export outside the source checkpoint directory.")
    source = actor / "model_world_size_1_rank_0.pt"
    source_info = source.stat()
    fsdp = json.loads((actor / "fsdp_config.json").read_text())
    original_config = json.loads((actor / "lora_adapter/adapter_config.json").read_text())
    model_config = json.loads((actor / "huggingface/config.json").read_text())
    torch.set_num_threads(1)
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    tensors, config, nonzero = extract_adapter(state, original_config, model_config, fsdp)
    output.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    actual = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    if actual.keys() != tensors.keys() or any(not torch.equal(tensors[k], actual[k]) for k in tensors):
        raise RuntimeError("Saved adapter does not exactly match the source tensors.")
    if source.stat().st_size != source_info.st_size or source.stat().st_mtime_ns != source_info.st_mtime_ns:
        raise RuntimeError("Source checkpoint changed during export.")
    report = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256(source),
        "source_size_bytes": source_info.st_size,
        "tensor_count": len(tensors),
        "nonzero_lora_B_tensor_count": nonzero,
        "all_adapter_tensors_finite": True,
        "safetensors_roundtrip_exact": True,
        "adapter_size_bytes": (output / "adapter_model.safetensors").stat().st_size,
        "adapter_sha256": sha256(output / "adapter_model.safetensors"),
        "base_model": config["base_model_name_or_path"],
        "base_model_revision": config["revision"],
        "base_model_revision_verified": False,
        "peft_reload_verified": False,
        "strategic_learning_demonstrated": False,
        "torch_version": torch.__version__,
        "source_adapter_config": original_config,
        "scope": "Tensor recovery only. Run verify_recovered_adapter.py for PEFT and forward equivalence.",
    }
    (output / "recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_adapter_config"}, indent=2))


if __name__ == "__main__":
    main()
