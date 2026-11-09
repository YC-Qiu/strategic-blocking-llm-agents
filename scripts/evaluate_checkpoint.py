"""Compare the frozen base and a pilot LoRA checkpoint through RAGEN rollouts.

Run in the existing Ubuntu RAGEN environment. No optimizer or training is used.
The recorded training configuration supplies the prompt and context protocol.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recorded_config(path):
    import yaml
    lines = path.read_text(encoding="utf-8").splitlines()
    # Hydra's --cfg output can contain import banners before the actual YAML.
    start = next(i for i, line in enumerate(lines) if line == "actor_rollout_ref:")
    result = yaml.safe_load("\n".join(lines[start:]))
    for key in ("custom_envs", "agent_proxy", "es_manager", "model_path"):
        if key not in result:
            raise ValueError(f"Recorded configuration is missing {key}.")
    return result


def evaluation_config(recorded, layouts, count, model_path, seed):
    config = deepcopy(recorded)
    template = config["custom_envs"]["BlockingWarmupRight"]
    if config["agent_proxy"]["enable_think"]:
        raise ValueError("This evaluation expects the action-only pilot.")
    tags = []
    for index, layout in enumerate(layouts):
        tag = f"PairedEval{index}"
        tags.append(tag)
        config["custom_envs"][tag] = deepcopy(template)
        config["custom_envs"][tag]["env_config"]["layout"] = layout
    config["es_manager"]["val"].update(
        env_groups=len(tags), group_size=count,
        env_configs={"tags": tags, "n_groups": [1] * len(tags)},
    )
    config["model_path"] = str(model_path)
    config["actor_rollout_ref"]["model"]["path"] = str(model_path)
    rollout = config["actor_rollout_ref"]["rollout"]
    rollout.update(max_num_seqs=4, seed=seed, enable_chunked_prefill=False)
    rollout["val_kwargs"].update(do_sample=True, temperature=1.0, top_p=1.0, top_k=-1)
    return config


def compare_base_tensor(saved, cached, allow_roundoff, rollout_dtype):
    """Audit tiny merge/unmerge drift without claiming exact checkpoint replay.

    Both evaluated policies use the same cached base. The absolute bound is only
    permitted for FP32 LoRA target weights; all other weights must match exactly.
    A narrow extension up to FP32 epsilon additionally requires exact equality
    of the whole tensor after conversion to the recorded rollout dtype.
    BF16 equality is reported separately, because tiny values can cross a BF16
    rounding boundary even when the FP32 difference is very small.
    """
    import torch
    if saved.shape != cached.shape:
        raise ValueError("Cached and checkpoint tensor shapes differ.")
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if rollout_dtype not in dtypes:
        raise ValueError(f"Unsupported explicit rollout dtype: {rollout_dtype}")
    target_dtype = dtypes[rollout_dtype]
    maximum = 0.0
    raw_different = 0
    rollout_different = 0
    left, right = saved.reshape(-1), cached.reshape(-1)
    for start in range(0, saved.numel(), 1_000_000):
        a, b = left[start:start + 1_000_000], right[start:start + 1_000_000]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Non-finite frozen base tensor.")
        raw_different += int((a != b).sum())
        maximum = max(maximum, (a.double() - b.double()).abs().max().item())
        cast_a, cast_b = a.to(target_dtype), b.to(target_dtype)
        if not torch.isfinite(cast_a).all() or not torch.isfinite(cast_b).all():
            raise ValueError("Non-finite base tensor after the rollout dtype conversion.")
        rollout_different += int((cast_a != cast_b).sum())
    eligible = allow_roundoff and saved.dtype == torch.float32
    limit = 1e-7 if eligible else 0.0
    # This is a fixed absolute cap, not a relative tolerance or a per-value ULP.
    extension = bool(eligible and maximum > limit
                     and maximum <= torch.finfo(torch.float32).eps
                     and rollout_different == 0)
    if extension:
        limit = torch.finfo(torch.float32).eps
    return {
        "checkpoint_dtype": str(saved.dtype), "cached_dtype": str(cached.dtype),
        "element_count": saved.numel(), "raw_different_elements": raw_different,
        "rollout_different_elements": rollout_different,
        "max_absolute_difference": maximum, "allowed_absolute_difference": limit,
        "rollout_exact_extension_used": extension,
        "accepted": maximum <= limit,
    }


def prepare_models(actor, output, rollout_dtype):
    """Verify the adapter exactly and audit the base used for adapter deployment."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from export_pilot_adapter import extract_adapter

    torch.set_num_threads(2)
    original = json.loads((actor / "lora_adapter/adapter_config.json").read_text())
    model_config = json.loads((actor / "huggingface/config.json").read_text())
    fsdp_config = json.loads((actor / "fsdp_config.json").read_text())
    model_id = original["base_model_name_or_path"]
    if model_id != "Qwen/Qwen2.5-0.5B-Instruct":
        raise ValueError(f"Unsupported pilot base: {model_id}")
    snapshot = Path(snapshot_download(model_id, revision=original.get("revision"), local_files_only=True))
    state = torch.load(actor / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=True, mmap=True)
    tensors, clean_config, _ = extract_adapter(state, original, model_config, fsdp_config)
    exported = load_file(str(actor / "lora_adapter/adapter_model.safetensors"))
    if exported.keys() != tensors.keys() or any(not torch.equal(exported[k], v) for k, v in tensors.items()):
        raise ValueError("Automatic adapter export differs from the full checkpoint.")

    frozen = {}
    for name, value in state.items():
        if "lora_" in name:
            continue
        if not name.startswith("base_model.model."):
            raise ValueError(f"Unexpected frozen tensor: {name}")
        key = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        if key in frozen:
            raise ValueError(f"Duplicate frozen tensor: {key}")
        frozen[key] = value
    checked = set()
    target_weights = {
        key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] + ".weight"
        for key in tensors
    }
    checks = {}
    shards = sorted(snapshot.glob("model*.safetensors"))
    if not shards:
        raise ValueError("No cached safetensors base weights found; no download was attempted.")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in checked or key not in frozen:
                    raise ValueError(f"Unexpected cached base tensor: {key}")
                checks[key] = compare_base_tensor(
                    frozen[key], handle.get_tensor(key), key in target_weights, rollout_dtype,
                )
                checked.add(key)
    missing = set(frozen) - checked
    if missing == {"lm_head.weight"} and model_config.get("tie_word_embeddings"):
        if "model.embed_tokens.weight" in checked and torch.equal(frozen["lm_head.weight"], frozen["model.embed_tokens.weight"]):
            checked.add("lm_head.weight")
    if checked != set(frozen):
        raise ValueError(f"Unverified frozen tensors: {sorted(set(frozen) - checked)}")
    failed = [key for key, item in checks.items() if not item["accepted"]]
    changed = [key for key, item in checks.items() if item["raw_different_elements"]]
    cast_changed = [key for key, item in checks.items() if item["rollout_different_elements"]]
    audit = {
        "policy": "Exact equality outside FP32 LoRA targets; target absolute drift <=1e-7, "
                  "or <=FP32 epsilon (1.1920928955078125e-7) only if the entire tensor "
                  "matches exactly after conversion to the recorded rollout dtype.",
        "rollout_exact_extension_tensors": [
            key for key, item in checks.items() if item["rollout_exact_extension_used"]
        ],
        "rollout_dtype": rollout_dtype, "failed_tensors": failed,
        "raw_different_tensors": changed, "rollout_different_tensors": cast_changed,
        "max_absolute_difference": max(item["max_absolute_difference"] for item in checks.values()),
        "tensors": checks,
    }
    write_json(output / "base-weight-audit.json", audit)
    if failed:
        raise ValueError(f"Base differences exceed the allowed roundoff policy: {failed[:3]}. "
                         f"Inspect {output / 'base-weight-audit.json'}; no evaluation was run.")
    if changed:
        print(f"Base audit: {len(changed)} tensors have bounded FP32 drift; "
              f"maximum absolute difference={audit['max_absolute_difference']:.9g}. "
              "Both policies will use the same cached base. This is exported-adapter evaluation, "
              "not exact full-checkpoint replay.", flush=True)
    adapter = output / "verified_adapter"
    adapter.mkdir()
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    write_json(adapter / "adapter_config.json", clean_config)
    report = {
        "snapshot": str(snapshot), "cached_revision": snapshot.name,
        "frozen_tensor_count": len(checked),
        "frozen_base_exactly_matches_checkpoint": not changed,
        "frozen_base_matches_at_rollout_dtype": not cast_changed,
        "base_roundoff_audit_passed": True,
        "same_cached_base_for_both_policies": True,
        "evaluation_scope": "Cached original base versus that same base plus the exported LoRA.",
        "full_checkpoint_replay_verified": False,
        "adapter_tensor_count": len(tensors), "export_matches_checkpoint": True,
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
        "source_actor": str(actor),
    }
    write_json(output / "model-verification.json", report)
    return snapshot, adapter


SEED_SCHEMES = ("sha256-v2", "legacy-linear-v1")


def request_seed(experiment_seed, env_id, turn, scheme="sha256-v2"):
    """Derive a stable request seed; both policies deliberately share it.

    Domain separation removes the old overlap between adjacent experiment seeds
    and turn numbers. Hash collisions remain theoretically possible in 63 bits.
    Environment IDs depend on the recorded layout order and episode count.
    """
    if min(experiment_seed, env_id, turn) < 0:
        raise ValueError("Seed, environment ID and turn must be nonnegative.")
    if scheme == "legacy-linear-v1":
        if turn >= 1000:
            raise ValueError("Legacy seed scheme supports fewer than 1000 turns.")
        return experiment_seed + env_id * 1000 + turn
    if scheme != "sha256-v2":
        raise ValueError(f"Unknown seed scheme: {scheme}")
    payload = f"strategic-blocking/sha256-v2/{experiment_seed}/{env_id}/{turn}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)


class SeededEngine:
    """Assign seeds by stable environment ID and turn, even as episodes finish."""

    def __init__(self, engine, adapter_request, seed, trace_path, seed_scheme="sha256-v2"):
        self.engine = engine
        self.adapter_request = adapter_request
        self.seed = seed
        self.seed_scheme = seed_scheme
        self.trace_path = trace_path
        self.env_ids = []
        self.turns = Counter()

    def generate(self, prompts, sampling_params):
        if len(prompts) != len(self.env_ids) or len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("Prompt/environment alignment failed.")
        params, records = [], []
        for env_id, prompt in zip(self.env_ids, prompts):
            turn = self.turns[env_id]
            if env_id < 0 or turn >= 1000:
                raise ValueError("Unsupported environment ID or turn count.")
            item = deepcopy(sampling_params)
            item.seed = request_seed(self.seed, env_id, turn, self.seed_scheme)
            params.append(item)
            records.append({"env_id": env_id, "turn": turn, "seed": item.seed,
                            "seed_scheme": self.seed_scheme,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "adapter_enabled": self.adapter_request is not None})
            self.turns[env_id] += 1
        outputs = self.engine.generate(prompts, sampling_params=params, lora_request=self.adapter_request)
        if len(outputs) != len(prompts):
            raise ValueError("vLLM returned an unexpected number of outputs.")
        active = set(self.engine.llm_engine.list_loras())
        expected = set() if self.adapter_request is None else {self.adapter_request.lora_int_id}
        if active != expected:
            raise ValueError(f"Unexpected registered adapters: {active}; expected {expected}.")
        with self.trace_path.open("a", encoding="utf-8") as stream:
            for record in records:
                record["registered_adapter_ids"] = sorted(active)
                stream.write(json.dumps(record) + "\n")
        return outputs


def worker(args):
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.ragen_dir))
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    import yaml
    config_dict = yaml.safe_load((args.output / "evaluation-config.yaml").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["system"]["CUDA_VISIBLE_DEVICES"])
    import ragen.llm_agent.agent_proxy as proxy_module
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest
    from verl import DataProto
    from probe_reward_signal import summarize_histories, json_default

    destination = args.output / args.worker
    destination.mkdir(exist_ok=False)
    config = OmegaConf.create(config_dict)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    request = None if args.worker == "base" else LoRARequest("pilot-final", 123, str(args.output / "verified_adapter"))
    original_llm = proxy_module.LLM

    def make_llm(*positional, **kwargs):
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=8,
                      seed=args.seed, max_num_seqs=4)
        return original_llm(*positional, **kwargs)

    class EvaluationWorker(proxy_module.VllmWrapperWg):
        def generate_sequences(self, lm_inputs):
            self.llm.env_ids = [int(x) for x in lm_inputs.non_tensor_batch["env_ids"]]
            return super().generate_sequences(lm_inputs)

    # Inject constructor settings only in this isolated evaluation process.
    proxy_module.LLM = make_llm
    try:
        actor = EvaluationWorker(config, tokenizer)
    finally:
        proxy_module.LLM = original_llm
    actor.llm = SeededEngine(actor.llm, request, args.seed, destination / "generation-trace.jsonl", args.seed_scheme)
    proxy = proxy_module.LLMAgentProxy(config, actor, tokenizer)
    rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False, "do_sample": True, "validate": True,
    }), val=True)
    rollouts.save_to_disk(str(destination / "rollout.pkl"))
    histories = list(rollouts.non_tensor_batch["history"])
    with (destination / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for index, history in enumerate(histories):
            stream.write(json.dumps({"episode_index": index, "history": history}, default=json_default) + "\n")
    summary = summarize_histories(histories, args.layouts, args.episodes_per_layout)
    write_json(destination / "summary.json", summary)
    print(f"Completed {args.worker}: {len(histories)} episodes.", flush=True)


def wilson(wins, total):
    z = 1.959963984540054
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def comparison(output, layouts, count):
    summaries = {name: json.loads((output / name / "summary.json").read_text()) for name in ("base", "adapter")}
    traces = {}
    for name in summaries:
        rows = [json.loads(x) for x in (output / name / "generation-trace.jsonl").read_text().splitlines()]
        traces[name] = {r["env_id"]: (r["seed"], r["prompt_sha256"], r.get("seed_scheme", "legacy-linear-v1")) for r in rows if r["turn"] == 0}
    if traces["base"] != traces["adapter"] or len(traces["base"]) != len(layouts) * count:
        raise ValueError("Initial prompts, sampling seeds or seed schemes differ across the comparison.")
    rows, action_distributions, cases = [], [], []
    for layout in layouts:
        for name, summary in summaries.items():
            episodes = [x for x in summary["episodes"] if x["layout"] == layout]
            if len(episodes) != count:
                raise ValueError("Incorrect episode count.")
            wins = sum(x["result"] == "A" for x in episodes)
            action_distributions.append({
                "layout": layout, "model": name,
                "executed_actions": dict(Counter(a for x in episodes for a in x["executed_actions"])),
                "first_executed_actions": dict(Counter(x["executed_actions"][0] for x in episodes if x["executed_actions"])),
            })
            for outcome in ("A", "B", "timeout", "unresolved"):
                example = next((x for x in episodes if x["result"] == outcome), None)
                if example is not None:
                    cases.append({"model": name, "layout": layout, "result": outcome,
                                  "episode_index": example["episode_index"],
                                  "raw_file": f"{name}/episodes.jsonl"})
            known = sum(x["known_action_validity_count"] for x in episodes)
            invalid = sum(x["invalid_parsed_action_count"] for x in episodes)
            low, high = wilson(wins, count)
            rows.append({"layout": layout, "model": name, "episodes": count,
                         "wins": wins, "success_rate": wins / count,
                         "success_ci95_low": low, "success_ci95_high": high,
                         "mean_observed_return": sum(x["observed_return"] for x in episodes) / count,
                         "unresolved": sum(not x["resolved"] for x in episodes),
                         "invalid_actions": invalid, "known_action_validity_count": known,
                         "invalid_action_rate": invalid / known if known else None,
                         "blocked_episodes": sum(x["block_count"] > 0 for x in episodes),
                         "format_errors": sum(x["format_error_count"] for x in episodes)})
    write_json(output / "case-index.json", cases)
    write_json(output / "comparison.json", {"initial_prompts_and_seeds_match": True, "rows": rows,
               "model_scope": "Same cached original base with and without exported LoRA; not exact full-checkpoint replay.",
               "action_distributions": action_distributions,
               "scope": "Fixed fixtures and one training seed; sampling uncertainty is not training-seed uncertainty.",
               "note": "An unresolved episode is not labeled a loss. Positive point estimates alone do not establish improvement."})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Base versus final LoRA evaluation", "", "| Layout | Model | Wins | Success rate | 95% Wilson interval | Unresolved |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['layout']} | {row['model']} | {row['wins']}/{count} | {row['success_rate']:.1%} | {row['success_ci95_low']:.1%}-{row['success_ci95_high']:.1%} | {row['unresolved']} |")
    lines.extend(["", "These are fixed fixtures, not a held-out map benchmark. The intervals describe sampling uncertainty for each fixed policy. Repeated training seeds and broader tests are needed for a robust improvement claim.", "", "Raw trajectories are in each model's episodes.jsonl. No explicit reasoning text is requested."])
    lines.extend(["", "Both policies use the same cached original base. This measures the exported LoRA's effect, not an exact replay of the full FP32 checkpoint. See base-weight-audit.json for checkpoint drift."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7 + len(rows)]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ragen-dir", type=Path, default=Path.home() / "RAGEN")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--episodes-per-layout", type=int, default=50)
    parser.add_argument("--layouts", nargs="+", default=["warmup_right", "warmup_left", "blocking_short"])
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--seed-scheme", choices=SEED_SCHEMES, default="sha256-v2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["base", "adapter"], help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_layout < 1 or args.seed < 0 or len(set(args.layouts)) != len(args.layouts):
        parser.error("Use a positive episode count, nonnegative seed and unique layouts.")
    args.project = args.project.resolve()
    args.ragen_dir = args.ragen_dir.expanduser().resolve()
    args.training_run = args.training_run.resolve()
    if args.worker:
        worker(args)
        return
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.project / "scripts"))
    from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv
    for layout in args.layouts:
        StrategicBlockingEnv(StrategicBlockingConfig(layout=layout)).close()
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output = Path(tempfile.mkdtemp(prefix="paired-eval-", dir=args.project / "results")).resolve()
    print(f"Evaluation output: {args.output}", flush=True)
    recorded = load_recorded_config(args.training_run / "composed.yaml")
    actor = args.training_run / f"checkpoints/global_step_{args.step}/actor"
    import yaml
    if args.prepare_only:
        config = evaluation_config(recorded, args.layouts, args.episodes_per_layout, recorded["model_path"], args.seed)
        (args.output / "evaluation-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print("Configuration prepared only; model verification and GPU execution have not run.")
        return
    print("Verifying frozen base and LoRA against the checkpoint on CPU...", flush=True)
    snapshot, adapter = prepare_models(actor, args.output, recorded["actor_rollout_ref"]["rollout"]["dtype"])
    gc.collect()
    config = evaluation_config(recorded, args.layouts, args.episodes_per_layout, snapshot, args.seed)
    (args.output / "evaluation-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_files = [args.project / name for name in ("maze.py", "ragen_maze.py", "curriculum.py", "frozen_eval.py", "evaluation/frozen-transfer-v1.json", "scripts/probe_reward_signal.py", "scripts/export_pilot_adapter.py")]
    source_files += [Path(__file__), args.training_run / "composed.yaml"]
    source_files += [args.ragen_dir / "ragen/llm_agent" / name for name in ("agent_proxy.py", "ctx_manager.py", "es_manager.py")]
    write_json(args.output / "manifest.json", {"training_run": str(args.training_run), "step": args.step,
               "layouts": args.layouts, "episodes_per_layout": args.episodes_per_layout,
               "seed": args.seed, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
               "seed_scheme": args.seed_scheme,
               "seed_rule": ("SHA256 domain-separated experiment seed, env_id and turn; first 8 bytes masked to 63 bits"
                             if args.seed_scheme == "sha256-v2" else "seed + env_id * 1000 + turn_index"),
               "updates_weights": False,
               "source_sha256": {str(path): file_hash(path) for path in source_files}})
    with (args.output / "dependencies.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(args.project), str(args.project / "scripts"), str(args.ragen_dir), environment.get("PYTHONPATH", "")])
    # Try adapter inference first, so a loading failure does not waste a baseline run.
    for name in ("adapter", "base"):
        log_path = args.output / f"{name}.log"
        print(f"Evaluating {name}: {len(args.layouts) * args.episodes_per_layout} episodes; log: {log_path}", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name,
                   "--project", str(args.project), "--ragen-dir", str(args.ragen_dir),
                   "--training-run", str(args.training_run), "--output", str(args.output),
                   "--seed", str(args.seed), "--seed-scheme", args.seed_scheme, "--episodes-per-layout", str(args.episodes_per_layout), "--layouts", *args.layouts]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=args.ragen_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Evaluation failed ({result.returncode}); inspect {log_path}. No training was run.")
    comparison(args.output, args.layouts, args.episodes_per_layout)
    print(f"Comparison complete: {args.output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
