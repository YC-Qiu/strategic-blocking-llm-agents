"""Evaluate frozen policies after a required, separately recorded format smoke test."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colab_protocol import RagenTextProtocol, read_protocol
from frozen_eval import load_suite, SUITE_SHA256
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv
from evaluate_checkpoint import wilson
from evaluate_colab_transfer import request_seed, verify_adapters

POLICIES = ("base", "before", "after")
SOURCE_FILES = ["maze.py", "demo.py", "curriculum.py", "ragen_maze.py", "frozen_eval.py",
                "colab_protocol.py", "scripts/evaluate_colab_transfer_v2.py",
                "scripts/evaluate_colab_transfer.py", "scripts/evaluate_checkpoint.py",
                "evaluation/frozen-transfer-v1.json", "evaluation/colab-protocol-v2.json",
                "evaluation/colab-ragen-contract-v2.json"]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_episode(layout, policy, index, generate, protocol, seed):
    """Generate returns a completion without the prompt's prefilled opening tag."""
    env = StrategicBlockingEnv(StrategicBlockingConfig(
        layout=layout, max_rounds=protocol["max_rounds"]))
    observation = env.reset()
    bridge = RagenTextProtocol(protocol)
    cache = [{"state": observation, "actions_left": protocol["max_actions_per_traj"]}]
    history, executed_count = [], 0
    for turn in range(protocol["max_manager_turns"]):
        if env.race.result is not None:
            break
        messages = bridge.messages(cache)
        sampled_seed = request_seed(seed, layout, index, turn)
        generated = generate(messages, sampled_seed)
        if isinstance(generated, str):
            generated = {"completion": generated}
        raw, normalized, actions = bridge.parse_completion(generated["completion"])
        format_error = not actions
        before = observation
        if format_error:
            reward, info = 0, {"executed_action": None, "action_is_valid": False}
        else:
            observation, reward, _, info = env.step(actions[0])
            executed_count += 1
        entry = {"observation": before, "next_observation": observation,
                 "completion": generated["completion"], "response": raw,
                 "normalized_response": normalized, "parsed_actions": actions,
                 "seed": sampled_seed, "manager_turn": turn + 1,
                 "environment_actions": executed_count,
                 "format_error": format_error, "manager_invalid_action": format_error,
                 "executed_action": info["executed_action"],
                 "action_is_valid": info["action_is_valid"], "reward": reward,
                 "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                 "generation": {k: v for k, v in generated.items() if k != "completion"}}
        # Retain complete prompts for the first two requests of each episode.
        if turn < 2:
            entry["messages"] = messages
        history.append(entry)
        cache[-1].update(llm_response=normalized, reward=reward, actions=actions)
        cache.append({"state": observation,
                      "actions_left": protocol["max_actions_per_traj"] - executed_count})
        if format_error:
            cache[-1]["manager_invalid_action"] = True
        if executed_count >= protocol["max_actions_per_traj"]:
            break
    return {"layout": layout, "policy": policy, "episode": index,
            "result": env.race.result or "unresolved", "history": history,
            "termination": "environment" if env.race.result is not None else "manager_budget",
            "final_observation": observation}


def format_gate(episodes, protocol):
    rows = {}
    spec = protocol["smoke"]
    expected = {(p, layout, i) for p in POLICIES for layout in spec["layouts"]
                for i in range(spec["episodes_per_map"])}
    actual = [(e["policy"], e["layout"], e["episode"]) for e in episodes]
    complete = len(actual) == len(expected) and set(actual) == expected
    for policy in POLICIES:
        turns = [t for e in episodes if e["policy"] == policy for t in e["history"]]
        errors = sum(t["format_error"] for t in turns)
        rate = errors / len(turns) if turns else 1.0
        rows[policy] = {"turns": len(turns), "format_errors": errors,
                        "format_error_rate": rate,
                        "passed": len(turns) >= spec["minimum_turns_per_policy"]
                        and rate <= spec["maximum_format_error_rate"]}
    return {"passed": complete and all(r["passed"] for r in rows.values()),
            "complete": complete, "policies": rows}


def verify_smoke(report_path, identity, protocol):
    directory = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path, episodes_path = directory / "manifest.json", directory / "episodes.jsonl"
    if not (directory / "COMPLETE").is_file() or (directory / "FAILED.json").exists():
        raise ValueError("Smoke test did not complete successfully.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("smoke_only") or manifest["identity"] != identity:
        raise ValueError("Smoke identity differs: rerun smoke with this code, tokenizer, adapters and runtime.")
    if report["manifest_sha256"] != file_hash(manifest_path) or report["episodes_sha256"] != file_hash(episodes_path):
        raise ValueError("Smoke evidence changed after the quality gate was recorded.")
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    recomputed = format_gate(episodes, protocol)
    if not recomputed["passed"] or any(report.get(k) != v for k, v in recomputed.items()):
        raise ValueError("Smoke format gate failed or its summary differs from raw episodes.")
    return {"report_sha256": file_hash(report_path), "report": report}


def summarize(episodes):
    rows = []
    for policy, layout in dict.fromkeys((e["policy"], e["layout"]) for e in episodes):
        selected = [e for e in episodes if e["policy"] == policy and e["layout"] == layout]
        turns = [t for e in selected for t in e["history"]]
        wins = sum(e["result"] == "A" for e in selected)
        n = len(selected)
        rows.append({"policy": policy, "layout": layout, "episodes": n, "wins": wins,
                     "success_rate": wins / n, "wilson95": wilson(wins, n),
                     "outcomes": {outcome: sum(e["result"] == outcome for e in selected)
                                  for outcome in ("A", "B", "timeout", "unresolved")},
                     "manager_turns": len(turns), "format_errors": sum(t["format_error"] for t in turns),
                     "illegal_actions": sum(not t["format_error"] and not t["action_is_valid"] for t in turns),
                     "executed_actions": sum(t["executed_action"] is not None for t in turns)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol, suite = read_protocol(), load_suite()
    verify_adapters(args.bundle, suite)
    if args.verify_only:
        print("Verified frozen maps, adapters, v2 protocol and RAGEN contract. No model loaded.")
        return
    if not args.smoke and args.smoke_report is None:
        parser.error("Full evaluation requires --smoke-report from a passing v2 smoke run.")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime.")
    tokenizer = AutoTokenizer.from_pretrained(suite["model_id"], revision=suite["base_revision"])
    # Verify that decoding cannot silently erase the protocol delimiters.
    tag_roundtrips = {tag: tokenizer.decode(tokenizer.encode(tag, add_special_tokens=False),
                                          skip_special_tokens=True, clean_up_tokenization_spaces=False)
                      for tag in ("<answer>", "</answer>")}
    if any(tag != decoded for tag, decoded in tag_roundtrips.items()):
        raise ValueError(f"Answer delimiters are lost during tokenization/decoding: {tag_roundtrips}")
    bridge = RagenTextProtocol(protocol)
    identity = {"suite_sha256": SUITE_SHA256, "base_revision": suite["base_revision"],
                "adapters": suite["policies"], "protocol": protocol,
                "sources": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "peft", "accelerate", "safetensors")},
                "gpu": torch.cuda.get_device_name(), "dtype": "float16", "attention": "sdpa",
                "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                "chat_template": tokenizer.chat_template, "tag_roundtrips": tag_roundtrips}
    smoke_evidence = None if args.smoke else verify_smoke(args.smoke_report, identity, protocol)
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="colab-smoke-v2-" if args.smoke else "colab-transfer-v2-",
                                  dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    layouts = protocol["smoke"]["layouts"] if args.smoke else [m["name"] for m in suite["maps"]]
    count = protocol["smoke"]["episodes_per_map"] if args.smoke else suite["episodes_per_map"]
    manifest = {"identity": identity, "smoke_only": args.smoke, "smoke_evidence": smoke_evidence,
                "layouts": layouts, "episodes_per_map": count, "experiment_seed": suite["experiment_seed"],
                "scope": protocol["backend"], "disclosure": protocol["disclosure"]}
    write_json(output / "manifest.json", manifest)
    (output / "dependencies.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    episodes = []
    try:
        base = AutoModelForCausalLM.from_pretrained(suite["model_id"], revision=suite["base_revision"],
                                                   torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")
        model = PeftModel.from_pretrained(base, args.bundle / "adapters/before",
                                         adapter_name="before", is_trainable=False)
        model.load_adapter(args.bundle / "adapters/after", adapter_name="after", is_trainable=False)
        model.eval()
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for policy in POLICIES:
                if policy != "base":
                    model.set_adapter(policy)
                model.eval()
                context = model.disable_adapter() if policy == "base" else nullcontext()
                with context, torch.inference_mode():
                    def generate(messages, seed):
                        prompt = bridge.prompt(tokenizer, messages)
                        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                        prompt_ids = inputs["input_ids"][0].tolist()
                        if len(prompt_ids) + protocol["max_new_tokens"] > protocol["max_context_tokens"]:
                            raise ValueError("Context budget exceeded; aborting without silent truncation.")
                        torch.manual_seed(seed)
                        ids = model.generate(**inputs, do_sample=True, **protocol["sampling"],
                                             max_new_tokens=protocol["max_new_tokens"],
                                             pad_token_id=tokenizer.eos_token_id)
                        completion_ids = ids[0, len(prompt_ids):].tolist()
                        return {"completion": tokenizer.decode(completion_ids, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False),
                                "decoded_with_special_tokens": tokenizer.decode(completion_ids, skip_special_tokens=False,
                                                                                clean_up_tokenization_spaces=False),
                                "completion_token_ids": completion_ids, "prompt_token_ids": prompt_ids,
                                "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                    for layout in layouts:
                        for index in range(count):
                            episode = run_episode(layout, policy, index, generate, protocol, suite["experiment_seed"])
                            stream.write(json.dumps(episode) + "\n")
                            stream.flush()
                            episodes.append(episode)
                            if not args.smoke:
                                turns = [t for e in episodes if e["policy"] == policy for t in e["history"]]
                                guard = protocol["full_run_guard"]
                                if len(turns) >= guard["minimum_turns_per_policy"] and sum(t["format_error"] for t in turns) / len(turns) > guard["maximum_format_error_rate"]:
                                    raise ValueError(f"Full-run format guard failed for {policy}; partial run retained.")
                        row = summarize(episodes)[-1]
                        print(f"{policy}: {layout}: {row['wins']}/{count}; format errors {row['format_errors']}/{row['manager_turns']}", flush=True)
        rows = summarize(episodes)
        write_json(output / "summary.json", rows)
        if args.smoke:
            gate = format_gate(episodes, protocol)
            gate.update(manifest_sha256=file_hash(output / "manifest.json"),
                        episodes_sha256=file_hash(output / "episodes.jsonl"))
            write_json(output / "quality-gate.json", gate)
            if not gate["passed"]:
                raise ValueError("Smoke format gate failed. Inspect raw outputs; do not start full evaluation.")
        lines = ["# Colab v2 evaluation", "", protocol["backend"], "", protocol["disclosure"], "",
                 "| Layout | Policy | Wins | Success | 95% Wilson interval | Format errors | Unresolved |",
                 "|---|---|---:|---:|---|---:|---:|"]
        for row in rows:
            lo, hi = row["wilson95"]
            lines.append(f"| {row['layout']} | {row['policy']} | {row['wins']}/{row['episodes']} | {row['success_rate']:.1%} | {lo:.1%}-{hi:.1%} | {row['format_errors']}/{row['manager_turns']} | {row['outcomes']['unresolved']} |")
        (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output / "COMPLETE").write_text("All declared episodes completed; required format checks passed.\n", encoding="utf-8")
        print(f"Complete: {output}", flush=True)
    except Exception as exc:
        write_json(output / "FAILED.json", {"error": str(exc), "completed_episodes": len(episodes)})
        write_json(output / "summary.json", summarize(episodes))
        raise


if __name__ == "__main__":
    main()
