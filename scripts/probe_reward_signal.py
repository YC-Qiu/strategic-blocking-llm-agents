"""Sample fixed maze groups through RAGEN without updating model weights.

Run in the existing Ubuntu RAGEN environment. The diagnostic grouping uses
identical initial observations; it does not assert anything about the trainer's
internal GRPO group indices. Output pickles must be local artifacts from this run.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv

DEFAULT_LAYOUTS = [
    "warmup_right", "warmup_left", "warmup_two_right", "warmup_two_left",
    "blocking_short", "slides", "costly",
]


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_config(project, output, layouts, group_size, seed, temperature):
    import yaml
    config = yaml.safe_load((project / "strategic_blocking.yaml").read_text())
    template = deepcopy(config["custom_envs"]["BlockingSlides"])
    template["env_instruction"] += (
        "\nYou choose A's action only. B has already moved. Copy one exact string "
        "from Legal actions, including its direction if present. Return it inside "
        "<answer>...</answer>. Use words, never an option number. Do not add an explanation."
    )
    tags = []
    for index, layout in enumerate(layouts):
        tag = f"Probe{index}"
        tags.append(tag)
        item = deepcopy(template)
        item["env_config"].update(layout=layout, max_rounds=32)
        config["custom_envs"][tag] = item
    config["model_path"] = "Qwen/Qwen2.5-0.5B-Instruct"
    config["agent_proxy"].update(enable_think=False, max_turn=32)
    config["es_manager"]["val"].update(
        env_groups=len(layouts), group_size=group_size,
        env_configs={"tags": tags, "n_groups": [1] * len(layouts)},
    )
    config["actor_rollout_ref"] = {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "max_model_len": 2048, "response_length": 128,
            "gpu_memory_utilization": 0.5, "max_num_seqs": 4,
            "max_num_batched_tokens": 2048, "enforce_eager": True,
            "temperature": temperature, "do_sample": True, "seed": seed,
            "val_kwargs": {
                "do_sample": True, "temperature": temperature,
                "top_p": 1.0, "top_k": -1,
            },
        }
    }
    config["output"].update(
        dir=str(output), filename="rollout.pkl", format="pkl",
        append_timestamp=False, save_pkl_backup=False, save_jsonl_backup=False,
    )
    return config


def episode_summary(history):
    """Keep manager truncation and unknown validity distinct from real losses."""
    history = list(history)
    turns = [row for row in history if "actions" in row]
    states = [row.get("state", "") for row in history]
    if not history or not isinstance(states[0], str):
        raise ValueError("Episode has no initial text observation.")
    outcomes = []
    for row in history:
        info = row.get("info") or {}
        if info.get("result") in ("A", "B", "timeout"):
            outcomes.append(info["result"])
        match = re.search(r"^Result: (A|B|timeout)$", row.get("state", ""), re.MULTILINE)
        if match:
            outcomes.append(match.group(1))
    if len(set(outcomes)) > 1:
        raise ValueError("Conflicting episode outcomes.")
    result = outcomes[-1] if outcomes else "unresolved"
    rewards = [float(row.get("reward", 0)) for row in turns]
    observed_return = sum(rewards)
    if result != "unresolved" and observed_return != {"A": 1, "B": -1, "timeout": 0}[result]:
        raise ValueError("Observed rewards disagree with the terminal outcome.")
    format_errors = 0
    invalid = 0
    known_validity = 0
    executed = []
    actions = []
    for row in turns:
        response = row.get("llm_raw_response", row.get("llm_response", ""))
        answers = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
        if len(answers) != 1 or not answers[0].strip():
            format_errors += 1
        info = row.get("info") or {}
        if "action_is_valid" in info:
            known_validity += 1
            invalid += not bool(info["action_is_valid"])
        actions.extend(str(action) for action in row.get("actions", []))
        if "executed_action" in info:
            executed.append(str(info["executed_action"]))
    return {
        "initial_observation_sha256": fingerprint(states[0]),
        "result": result, "resolved": result != "unresolved",
        "observed_return": observed_return,
        "turn_count": len(turns),
        "format_error_count": format_errors,
        "known_action_validity_count": known_validity,
        "invalid_parsed_action_count": invalid,
        "parsed_actions": actions, "executed_actions": executed,
        "block_count": sum(a.startswith("block ") for a in executed),
        "wait_count": executed.count("wait"),
    }


def summarize_histories(histories, layouts, group_size):
    expected = {}
    for layout in layouts:
        env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
        initial = env.reset()
        env.close()
        key = fingerprint(initial)
        if key in expected:
            raise ValueError("Diagnostic layouts have duplicate initial observations.")
        expected[key] = layout
    groups = defaultdict(list)
    episodes = []
    for index, history in enumerate(histories):
        item = episode_summary(history)
        key = item["initial_observation_sha256"]
        if key not in expected:
            raise ValueError("Unknown initial observation; inspect raw records before regrouping.")
        item.update(episode_index=index, layout=expected[key])
        episodes.append(item)
        groups[item["layout"]].append(item)
    summaries = []
    for layout in layouts:
        group = groups[layout]
        if len(group) != group_size:
            raise ValueError(f"{layout}: expected {group_size} episodes, found {len(group)}.")
        resolved = all(item["resolved"] for item in group)
        returns = [item["observed_return"] for item in group]
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        summaries.append({
            "layout": layout, "episode_count": len(group),
            "outcome_counts": dict(Counter(item["result"] for item in group)),
            "returns": returns, "all_episodes_resolved": resolved,
            "reward_variance": variance,
            "informative_terminal_group": resolved and variance > 0,
            "format_errors": sum(item["format_error_count"] for item in group),
            "invalid_parsed_actions": sum(item["invalid_parsed_action_count"] for item in group),
            "known_action_validity_count": sum(item["known_action_validity_count"] for item in group),
            "blocked_episodes": sum(item["block_count"] > 0 for item in group),
        })
    return {
        "scope": "Exploratory sampled evaluation; no training or parameter updates.",
        "grouping": "One diagnostic group per identical initial observation.",
        "training_grpo_group_alignment_verified": False,
        "group_size": group_size, "episode_count": len(episodes),
        "informative_group_count": sum(g["informative_terminal_group"] for g in summaries),
        "group_count": len(summaries), "groups": summaries,
        "episodes": episodes,
    }


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ragen-dir", type=Path, default=Path.home() / "RAGEN")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.group_size < 2 or args.temperature <= 0:
        parser.error("Use group-size >= 2 and temperature > 0 for the sampled probe.")
    if len(set(args.layouts)) != len(args.layouts):
        parser.error("Layouts must be unique.")
    project = args.project.resolve()
    for layout in args.layouts:
        StrategicBlockingEnv(StrategicBlockingConfig(layout=layout)).close()
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (project / "results").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="a3-probe-", dir=project / "results")).resolve()
    import yaml
    config = make_config(project, output, args.layouts, args.group_size, args.seed, args.temperature)
    config_text = yaml.safe_dump(config, sort_keys=False)
    (output / "config.yaml").write_text(config_text)
    manifest = {
        "layouts": args.layouts, "episodes_per_layout": args.group_size,
        "expected_episodes": len(args.layouts) * args.group_size,
        "sampling_seed_requested": args.seed, "temperature": args.temperature,
        "action_only": True, "model": config["model_path"],
        "reward_shaping": False, "updates_weights": False,
        "source_sha256": {name: fingerprint((project/name).read_text()) for name in
                          ("maze.py", "ragen_maze.py", "curriculum.py", "strategic_blocking.yaml",
                           "scripts/probe_reward_signal.py")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Probe output: {output}", flush=True)
    if args.prepare_only:
        print("Configuration prepared; no RAGEN or model process started.", flush=True)
        return
    ragen = args.ragen_dir.expanduser().resolve()
    if not (ragen / "train.py").is_file():
        raise ValueError("RAGEN checkout not found; pass --ragen-dir.")
    config_handle = tempfile.NamedTemporaryFile(
        mode="w", prefix="strategic_blocking_probe_", suffix=".yaml",
        dir=ragen / "config", delete=False,
    )
    with config_handle:
        config_handle.write(config_text)
    config_path = Path(config_handle.name)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    command = [
        sys.executable, "-u", "-m", "ragen.llm_agent.agent_proxy",
        "--config-path", str(ragen / "config"), "--config-name", config_path.stem,
    ]
    # Preserve the exact installed versions alongside the raw rollout.
    with (output / "dependencies.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    for name, arguments in (("ragen-revision.txt", ["rev-parse", "HEAD"]),
                            ("ragen-working-tree.patch", ["diff", "HEAD"]),
                            ("ragen-status.txt", ["status", "--short"])):
        with (output / name).open("w") as stream:
            subprocess.run(["git", "-C", str(ragen), *arguments], stdout=stream, check=True)
    print(f"Sampling {manifest['expected_episodes']} episodes; log: {output / 'run.log'}", flush=True)
    with (output / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ragen, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"RAGEN probe failed ({result.returncode}); inspect {output / 'run.log'}")
    # Read only the pickle generated by this invocation.
    import pickle
    sys.path.insert(0, str(ragen))
    with (output / "rollout.pkl").open("rb") as stream:
        data = pickle.load(stream)
    batch = data.non_tensor_batch
    histories = list(batch["history"])
    with (output / "episodes.jsonl").open("w") as stream:
        for index, history in enumerate(histories):
            record = {"episode_index": index, "history": history}
            for field in ("messages_list", "metrics"):
                if field in batch and len(batch[field]) == len(histories):
                    record[field] = batch[field][index]
            stream.write(json.dumps(record, default=json_default) + "\n")
    # Export raw JSON before strict consistency checks, so failures are inspectable.
    summary = summarize_histories(histories, args.layouts, args.group_size)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for group in summary["groups"]:
        print(f"{group['layout']}: returns={group['returns']}; informative={group['informative_terminal_group']}", flush=True)
    print(f"Informative diagnostic groups: {summary['informative_group_count']}/{summary['group_count']}", flush=True)
    print(f"Inspect raw trajectories before choosing a training curriculum: {output}", flush=True)


if __name__ == "__main__":
    main()
