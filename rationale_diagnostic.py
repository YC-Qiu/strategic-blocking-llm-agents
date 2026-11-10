"""Fixed-state explanation diagnostics, separate from the primary policy evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from colab_protocol import RagenTextProtocol, read_protocol
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv

ROOT = Path(__file__).resolve().parent


def specification():
    return json.loads((ROOT / "evaluation/rationale-diagnostic-v1.json").read_text(encoding="utf-8"))


def state_for(fixture):
    env = StrategicBlockingEnv(StrategicBlockingConfig(layout=fixture["layout"], max_rounds=32))
    env.reset()
    for action in fixture["prefix_actions"]:
        if action not in env.race.legal_actions():
            raise ValueError("Fixture prefix contains an illegal action.")
        env.step(action)
    if env.race.result is not None:
        raise ValueError("Diagnostic fixture must be an active decision state.")
    return env


def messages_for(fixture, condition):
    env = state_for(fixture)
    protocol = read_protocol()
    bridge = RagenTextProtocol(protocol)
    # Every snapshot is an independent request. No selected model response is supplied.
    messages = bridge.messages([{"state": env.render(), "actions_left": 32-env.race.a_actions}])
    if condition == "action_only":
        return messages, "<answer>"
    if condition != "brief_rationale":
        raise ValueError("Unknown diagnostic condition.")
    messages[0]["content"] = messages[0]["content"].replace(
        "Do not add an explanation.",
        "Before the action, provide a brief decision explanation in one or two sentences inside <rationale>...</rationale>. "
        "State the action you intend to take and why it helps A in the displayed state. "
        "Then provide exactly one legal action inside <answer>...</answer>. Do not write anything else.")
    messages[1]["content"] = messages[1]["content"].replace(
        "Always output: <answer> [your answer] </answer> with no extra text.",
        "Always output: <rationale> [one or two sentences explaining the chosen action] </rationale><answer> [your answer] </answer> with no extra text.")
    return messages, "<rationale>"


def parse(completion, condition):
    if condition == "action_only":
        raw, normalized, actions = RagenTextProtocol(read_protocol()).parse_completion(completion)
        return {"response": raw, "normalized_response": normalized, "rationale": None,
                "action": actions[0] if actions else None, "format_ok": bool(actions)}
    if condition != "brief_rationale":
        raise ValueError("Unknown diagnostic condition.")
    raw = "<rationale>" + completion
    match = re.fullmatch(r"\s*<rationale>([^<>]+)</rationale>\s*<answer>([^<>]+)</answer>\s*", raw, re.DOTALL)
    valid = bool(match and match[1].strip() and match[2].strip() and "||" not in match[2])
    return {"response": raw, "normalized_response": None,
            "rationale": match[1].strip() if valid else None,
            "action": match[2].strip() if valid else None, "format_ok": valid}


def seed_for(fixture_id, repetition):
    # Same request seed across policies and conditions; different prompts change distributions.
    key = f"rationale-diagnostic-v1/20261118/{fixture_id}/{repetition}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63)-1)


def distances(race):
    return {name: len(race.maze.path(start, goal, race.block))-1
            for name, start, goal in [("A", race.a, race.ga), ("B", race.b, race.gb)]}


def facts_for(fixture):
    env = state_for(fixture)
    result = {"observation": env.render(), "state": env.race.observe(),
              "shortest_distances_before_action": distances(env.race), "legal_action_effects": {}}
    for action in env.race.legal_actions():
        candidate = deepcopy(env)
        candidate.race.act(action)
        after_a = {"state": candidate.race.observe(), "distances": distances(candidate.race)}
        if candidate.race.result is None:
            candidate.race.advance_b()
        result["legal_action_effects"][action] = {"after_A_before_B": after_a,
            "after_next_B_if_active": candidate.render(), "result": candidate.race.result}
    return result


def summarize(rows, spec):
    expected = {(p, c, f["id"], i) for p in spec["policies"] for c in spec["conditions"]
                for f in spec["fixtures"] for i in range(spec["repetitions"])}
    keys = [(r["policy"], r["condition"], r["fixture_id"], r["repetition"]) for r in rows]
    groups = []
    for policy in spec["policies"]:
        for condition in spec["conditions"]:
            selected = [r for r in rows if r["policy"] == policy and r["condition"] == condition]
            groups.append({"policy": policy, "condition": condition, "requests": len(selected),
                           "format_valid": sum(r["format_ok"] for r in selected),
                           "legal_action": sum(r["action_is_legal"] for r in selected),
                           "token_budget_exhausted": sum(r["token_budget_exhausted"] for r in selected)})
    changes = []
    indexed = dict(zip(keys, rows))
    for policy in spec["policies"]:
        comparable = changed = 0
        for f in spec["fixtures"]:
            for i in range(spec["repetitions"]):
                a = indexed.get((policy, "action_only", f["id"], i))
                b = indexed.get((policy, "brief_rationale", f["id"], i))
                if a and b and a["format_ok"] and b["format_ok"] and a["action_is_legal"] and b["action_is_legal"]:
                    comparable += 1
                    changed += a["action"].lower() != b["action"].lower()
        changes.append({"policy": policy, "both_valid_pairs": comparable, "different_actions": changed})
    return {"complete": len(keys) == len(expected) and set(keys) == expected,
            "expected_requests": len(expected), "groups": groups, "action_changes": changes,
            "semantic_review": "PENDING: complete the manual review CSV; format and legality do not establish factual or action-explanation consistency.",
            "scope": "Fixed-state elicited explanations, not episodes, primary win-rate evaluation, or verified internal reasoning."}
