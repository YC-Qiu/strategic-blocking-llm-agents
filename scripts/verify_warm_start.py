"""Validate the pilot adapter and verify its first collected training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect_adapter(adapter):
    import torch
    from safetensors.torch import load_file
    from pilot_lora_collection import _pilot_extract_named_lora
    adapter = adapter.resolve()
    config = json.loads((adapter / "adapter_config.json").read_text())
    required = {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Expected the existing Qwen2.5-0.5B rank-8 pilot adapter.")
    for key in ("use_dora", "use_rslora", "fan_in_fan_out", "lora_bias", "modules_to_save", "rank_pattern", "alpha_pattern", "layer_replication", "target_parameters", "trainable_token_indices", "use_qalora", "alora_invocation_tokens", "use_bdlora"):
        if config.get(key):
            raise ValueError(f"Unsupported adapter setting: {key}")
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    names = [(name.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"), value) for name, value in tensors.items()]
    checked = _pilot_extract_named_lora(names, layers=24, rank=8)
    if any(value.dtype != torch.float32 for value in checked.values()):
        raise ValueError("Expected the verified FP32 pilot adapter export.")
    digest = hashlib.sha256()
    for name, value in sorted(checked.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return {"adapter_path": str(adapter), "tensor_count": len(checked), "tensor_sha256": digest.hexdigest(), "optimizer_restored": False, "scope": "Adapter warm start with a fresh optimizer; not an exact training resume."}


def verify_initial_collection(manifest, log):
    if "Loading pre-trained LoRA adapter to actor from:" not in log:
        raise ValueError("Trainer did not report loading the supplied actor adapter.")
    match = re.search(r"PILOT_LORA_COLLECT tensors=(\d+) sha256=([0-9a-f]{64})", log)
    if not match or int(match[1]) != manifest["tensor_count"] or match[2] != manifest["tensor_sha256"]:
        raise ValueError("Initial collected weights do not match the requested warm-start adapter.")
    return {"first_collected_weights_match_source": True, "tensor_sha256": match[2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "warm-start.json").read_text())
    log = (args.run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    result = verify_initial_collection(manifest, log)
    (args.run_dir / "warm-start-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("PASS: first collected actor weights match the supplied warm-start adapter.")


if __name__ == "__main__":
    main()
