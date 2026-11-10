"""Run 72 independent fixed-state requests; preserve every output for manual review."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rationale_diagnostic import facts_for, messages_for, parse, seed_for, specification, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=ROOT / "results/paired-eval-trum2mve/verified_adapter")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "results/rationale-diagnostic")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    spec = specification()
    for name, key in [("adapter_model.safetensors", "adapter_weights_sha256"), ("adapter_config.json", "adapter_config_sha256")]:
        if sha(args.adapter / name) != spec[key]:
            raise ValueError(f"Wrong final blocking adapter: {name}")
    facts = {f["id"]: facts_for(f) for f in spec["fixtures"]}
    for fixture in spec["fixtures"]:
        for condition in spec["conditions"]:
            messages_for(fixture, condition)
    if args.check_only:
        print("PASS: six valid decision states and final adapter hashes. No model loaded or generation run.")
        return

    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="rationale-v1-", dir=args.output_parent))
    print(f"Diagnostic output: {output}", flush=True)
    rows = []
    write_json(output / "protocol.json", spec)
    write_json(output / "state-facts.json", facts)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required. No training will be run.")
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["base_revision"], local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["base_revision"],
            local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")
        model = PeftModel.from_pretrained(base, args.adapter, adapter_name="after", is_trainable=False)
        model.eval()
        sources = ["rationale_diagnostic.py", "scripts/run_rationale_diagnostic.py", "evaluation/rationale-diagnostic-v1.json",
                   "colab_protocol.py", "maze.py", "ragen_maze.py", "demo.py", "curriculum.py", "frozen_eval.py",
                   "evaluation/frozen-transfer-v1.json", "evaluation/colab-protocol-v2.json", "evaluation/colab-ragen-contract-v2.json"]
        write_json(output / "manifest.json", {"protocol": spec, "source_sha256": {n: sha(ROOT / n) for n in sources},
            "adapter": str(args.adapter.resolve()), "gpu": torch.cuda.get_device_name(), "dtype": "float16", "attention": "sdpa",
            "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "safetensors"]},
            "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "chat_template": tokenizer.chat_template, "generation_budget_per_condition": spec["max_new_tokens"]})
        with (output / "responses.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for policy in spec["policies"]:
                if policy == "after":
                    model.set_adapter("after")
                context = model.disable_adapter() if policy == "base" else nullcontext()
                with context, torch.inference_mode():
                    for condition in spec["conditions"]:
                        for fixture in spec["fixtures"]:
                            messages, prefix = messages_for(fixture, condition)
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
                            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                            prompt_ids = tokens["input_ids"][0].tolist()
                            if len(prompt_ids) + spec["max_new_tokens"] > spec["max_context_tokens"]:
                                raise ValueError("Context limit exceeded; no silent truncation.")
                            for repetition in range(spec["repetitions"]):
                                seed = seed_for(fixture["id"], repetition)
                                torch.manual_seed(seed)
                                generated = model.generate(**tokens, do_sample=True, **spec["sampling"],
                                    max_new_tokens=spec["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
                                completion_ids = generated[0, len(prompt_ids):].tolist()
                                completion = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                parsed = parse(completion, condition)
                                legal = bool(parsed["format_ok"] and parsed["action"].lower() in facts[fixture["id"]]["state"]["legal_actions"])
                                eos = model.generation_config.eos_token_id
                                eos_ids = set(eos if isinstance(eos, list) else [eos])
                                row = {"policy": policy, "condition": condition, "fixture_id": fixture["id"],
                                    "repetition": repetition, "seed": seed, "messages": messages, "assistant_prefix": prefix,
                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_token_ids": prompt_ids,
                                    "completion": completion, "completion_token_ids": completion_ids,
                                    "decoded_with_special_tokens": tokenizer.decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                                    **parsed, "action_is_legal": legal,
                                    "token_budget_exhausted": len(completion_ids) >= spec["max_new_tokens"] and completion_ids[-1] not in eos_ids}
                                stream.write(json.dumps(row) + "\n")
                                stream.flush()
                                rows.append(row)
                            print(f"{len(rows)}/72: {policy}, {condition}, {fixture['id']}", flush=True)
        summary = summarize(rows, spec)
        write_json(output / "summary.json", summary)
        if not summary["complete"]:
            raise ValueError("Missing or duplicate requests; output retained.")
        columns = ["policy", "condition", "fixture_id", "repetition", "action", "rationale", "format_ok", "action_is_legal",
                   "factual_correctness", "rationale_action_agreement", "strategic_justification", "evidence_and_notes"]
        with (output / "manual-review.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                record = {k: row.get(k, "") for k in columns}
                status = "not_reviewed" if row["rationale"] else "not_assessable"
                record.update({k: status for k in ["factual_correctness", "rationale_action_agreement", "strategic_justification"]})
                writer.writerow(record)
        lines = ["# Fixed-state rationale diagnostic", "", summary["scope"], "", summary["semantic_review"], "",
                 "| Policy | Condition | Requests | Format valid | Legal action | Token budget exhausted |",
                 "|---|---|---:|---:|---:|---:|"]
        for row in summary["groups"]:
            lines.append(f"| {row['policy']} | {row['condition']} | {row['requests']} | {row['format_valid']} | {row['legal_action']} | {row['token_budget_exhausted']} |")
        lines += ["", "Action changes compare only pairs with legal, format-valid outputs in both conditions. They do not measure win rate or explanation faithfulness.",
                  "", json.dumps(summary["action_changes"], indent=2), "",
                  "Use state-facts.json to inspect factual claims and action effects. Review all rationale responses before selecting examples. Preserve invalid and truncated outputs in the denominators."]
        (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
        (output / "COMPLETE_GENERATION").write_text("All 72 responses recorded. Semantic review is still pending.\n", encoding="utf-8")
        print(f"Generation complete; semantic review pending: {output}", flush=True)
    except Exception:
        write_json(output / "FAILED.json", {"completed_requests": len(rows), "traceback": traceback.format_exc()})
        write_json(output / "summary.json", summarize(rows, spec))
        raise


if __name__ == "__main__":
    main()
