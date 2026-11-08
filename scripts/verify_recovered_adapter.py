"""Verify recovered LoRA tensors and CPU inference against the saved checkpoint.

Uses the frozen base weights embedded in the checkpoint, so no model download is
needed. This checks export equivalence, not strategic learning or upstream model
revision identity. Run in the working Ubuntu environment with PEFT installed.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

from export_pilot_adapter import extract_adapter, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; choose a fresh report path.")
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(f"Run in the working RAGEN Python environment: {error}") from error

    torch.set_num_threads(2)
    torch.manual_seed(0)
    actor = args.actor_dir.resolve()
    adapter = args.adapter_dir.resolve()
    source = actor / "model_world_size_1_rank_0.pt"
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    original_config = json.loads((actor / "lora_adapter/adapter_config.json").read_text())
    model_config = json.loads((actor / "huggingface/config.json").read_text())
    fsdp_config = json.loads((actor / "fsdp_config.json").read_text())
    expected, expected_config, _ = extract_adapter(state, original_config, model_config, fsdp_config)
    if json.loads((adapter / "adapter_config.json").read_text()) != expected_config:
        raise ValueError("Recovered adapter configuration differs from the supported source settings.")
    disk = load_file(str(adapter / "adapter_model.safetensors"))
    if expected.keys() != disk.keys() or any(not torch.equal(expected[k], disk[k]) for k in expected):
        raise ValueError("Recovered tensors do not exactly match the source checkpoint.")
    print("PASS: all saved adapter tensors match the source.", flush=True)

    def new_base():
        config = AutoConfig.from_pretrained(actor / "huggingface", local_files_only=True)
        return AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").float()

    reference = get_peft_model(new_base(), LoraConfig.from_pretrained(adapter))
    reference.load_state_dict(state, strict=True, assign=True)
    reference.eval()
    reference_adapter = get_peft_model_state_dict(reference)
    if reference_adapter.keys() != expected.keys() or any(
        not torch.equal(reference_adapter[k].cpu(), expected[k]) for k in expected
    ):
        raise ValueError("PEFT's own key mapping disagrees with the recovered export.")

    sys.path.insert(0, str(args.project.resolve()))
    from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv

    tokenizer = AutoTokenizer.from_pretrained(actor / "huggingface", local_files_only=True)
    fixtures = []
    with torch.inference_mode():
        for layout in ("slides", "costly"):
            env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
            observation = env.reset()
            env.close()
            messages = [
                {"role": "system", "content": "Control A. Return one legal action inside <answer>...</answer>."},
                {"role": "user", "content": observation},
            ]
            batch = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            logits = reference(**batch, use_cache=False).logits[:, -1, :].clone()
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite reference logits.")
            fixtures.append((layout, batch, logits))
    del reference, reference_adapter
    gc.collect()
    print("PASS: reference model loaded strictly and produced finite fixture logits.", flush=True)

    base_state = {}
    for key, value in state.items():
        if "lora_" in key:
            continue
        if not key.startswith("base_model.model."):
            raise ValueError(f"Unexpected base-model key: {key}")
        mapped = key.removeprefix("base_model.model.").replace(".base_layer.", ".")
        if mapped in base_state:
            raise ValueError(f"Duplicate base-model key: {mapped}")
        base_state[mapped] = value
    base = new_base()
    base.load_state_dict(base_state, strict=True, assign=True)
    candidate = PeftModel.from_pretrained(
        base, adapter, is_trainable=False, autocast_adapter_dtype=False,
    ).eval()
    reloaded = get_peft_model_state_dict(candidate)
    if reloaded.keys() != expected.keys() or any(
        not torch.equal(reloaded[k].cpu(), expected[k]) for k in expected
    ):
        raise ValueError("PEFT reloaded tensors differ from the source.")

    comparisons = []
    with torch.inference_mode():
        for layout, batch, before in fixtures:
            after = candidate(**batch, use_cache=False).logits[:, -1, :]
            torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
            comparisons.append({
                "layout": layout,
                "max_abs_logit_difference": (after - before).abs().max().item(),
                "same_greedy_next_token": bool(torch.equal(after.argmax(-1), before.argmax(-1))),
            })
        batch = fixtures[0][1]
        generated = candidate.generate(
            **batch, do_sample=False, max_new_tokens=24,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            generated[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
    report = {
        "peft_reload_verified": True,
        "peft_tensor_mapping_exact": True,
        "full_checkpoint_reference_loaded_strictly": True,
        "base_weights_source": "Frozen base tensors from the original full checkpoint",
        "upstream_base_revision_verified": False,
        "backend": "CPU float32 SDPA",
        "fixture_scope": "Two fixed maze prompts; not the full RAGEN evaluation protocol",
        "comparisons": comparisons,
        "generation_smoke_test_response": response,
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "strategic_learning_demonstrated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
