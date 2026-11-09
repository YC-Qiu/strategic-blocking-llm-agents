"""Replay local evaluation actions and ablate the first successful block.

The counterfactual holds later recorded actions fixed. It does not query the
policy again and cannot reveal an internal reasoning process.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from curriculum import CURRICULUM_MAPS
from frozen_eval import FROZEN_MAPS
REPLAY_MAPS = {**CURRICULUM_MAPS, **FROZEN_MAPS}
from maze import Maze, Race


def validate_manager_rejection(step, following):
    """A manager format rejection consumes no environment action or race turn."""
    info = step.get("info", {})
    if (info.get("manager_invalid_action") is not True
            or "executed_action" in info or step.get("actions") != []
            or info.get("raw_reward") != 0.0 or step.get("reward") != 0.0
            or not step.get("state") or following is None
            or step["state"] != following.get("state")):
        raise ValueError("Unverified manager rejection; cannot skip this record.")


def audit_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    import yaml
    config = yaml.safe_load((root / "evaluation-config.yaml").read_text())
    limits = {item["env_config"]["layout"]: item["env_config"]["max_rounds"]
              for tag, item in config["custom_envs"].items() if tag.startswith("PairedEval")}
    if set(limits) != set(manifest["layouts"]):
        raise ValueError("Evaluation configuration and manifest layouts differ.")
    report = {"run": root.name, "seed": manifest["seed"],
              "seed_scheme": manifest.get("seed_scheme", "legacy-linear-v1"),
              "policies": {}, "source_sha256": {}}
    for policy in ("base", "adapter"):
        summary_path = root / policy / "summary.json"
        raw_path = root / policy / "episodes.jsonl"
        episodes = json.loads(summary_path.read_text())["episodes"]
        lookup = {episode["episode_index"]: episode for episode in episodes}
        if len(lookup) != len(episodes):
            raise ValueError("Duplicate episode indices in summary.")
        seen, blocking_cases = set(), []
        manager_rejections = 0
        for line in raw_path.read_text().splitlines():
            record = json.loads(line)
            index = record["episode_index"]
            if index in seen or index not in lookup:
                raise ValueError("Duplicate or unexpected raw episode index.")
            seen.add(index)
            expected = lookup[index]
            layout = expected["layout"]
            race = Race(Maze(REPLAY_MAPS[layout]), limits[layout])
            race.advance_b()
            actions = []
            for offset, step in enumerate(record["history"]):
                if "info" not in step:
                    continue
                info = step["info"]
                if "executed_action" not in info:
                    following = record["history"][offset + 1] if offset + 1 < len(record["history"]) else None
                    validate_manager_rejection(step, following)
                    manager_rejections += 1
                    continue
                action = info["executed_action"]
                race.act(action)
                actions.append(action)
                if race.result is None:
                    race.advance_b()
                if race.result != info["result"] or race.reward != info["raw_reward"]:
                    raise ValueError(f"Replay mismatch: {policy}, episode {index}.")
            if race.result != expected["result"] or actions != expected["executed_actions"]:
                raise ValueError(f"Summary mismatch: {policy}, episode {index}.")
            if layout != "blocking_short" or race.result != "A":
                continue
            counterfactual = Race(Maze(REPLAY_MAPS[layout]), limits[layout])
            counterfactual.advance_b()
            removed = False
            for action in actions:
                if counterfactual.result is not None:
                    break
                if action.startswith("block ") and not removed:
                    action, removed = "wait", True
                counterfactual.act(action if action in counterfactual.legal_actions() else "wait")
                if counterfactual.result is None:
                    counterfactual.advance_b()
            blocking_cases.append({"episode_index": index, "actions": actions,
                                   "first_block_removed": removed,
                                   "counterfactual_result": counterfactual.result,
                                   "counterfactual_a_actions": counterfactual.a_actions})
        if seen != set(lookup):
            raise ValueError("Some summary episodes have no raw trajectory.")
        counts = Counter(episode["layout"] for episode in episodes)
        if dict(counts) != {layout: manifest["episodes_per_layout"] for layout in manifest["layouts"]}:
            raise ValueError("Episode counts differ from the evaluation manifest.")
        report["policies"][policy] = {"replayed_episodes": len(seen), "verified_manager_rejections": manager_rejections, "blocking_cases": blocking_cases}
        for path in (summary_path, raw_path):
            report["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.evaluation_run.resolve())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Replayed {sum(p['replayed_episodes'] for p in result['policies'].values())} episodes.")
    print(f"Audit: {args.output}")
