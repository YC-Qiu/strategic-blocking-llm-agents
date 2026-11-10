"""Regression checks for prompt parity, state preservation and smoke gating."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
from colab_protocol import RagenTextProtocol, read_protocol
from evaluate_colab_transfer_v2 import (run_episode, format_gate, verify_smoke,
                                        file_hash, write_json, POLICIES)


class ColabV2Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_protocol()
        self.bridge = RagenTextProtocol(self.protocol)

    def test_prefill_and_restoration_require_a_generated_closing_tag(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "CHAT_ASSISTANT\n"
        self.assertEqual(self.bridge.prompt(Tokenizer(), []), "CHAT_ASSISTANT\n<answer>")
        self.assertEqual(self.bridge.parse_completion("move right</answer>")[2], ["move right"])
        self.assertEqual(self.bridge.parse_completion("move right")[2], [])

    def test_captured_parser_retains_ragen_first_action_behavior(self):
        _, normalized, actions = self.bridge.parse_completion("move right || wait</answer>ignored")
        self.assertEqual(actions, ["move right"])
        self.assertEqual(normalized, "<answer>move right</answer>")
        self.assertEqual(self.bridge.parse_completion("</answer>")[2], [])

    def test_window_includes_current_state_and_preserves_absolute_turns(self):
        cache = [{"state": str(i), "actions_left": 32-i,
                  "llm_response": "<answer>wait</answer>", "reward": 0} for i in range(3)]
        cache.append({"state": "current", "actions_left": 29, "manager_invalid_action": True})
        messages = self.bridge.messages(cache)
        self.assertEqual(len(messages), 4)
        self.assertIn("Turn 3:\nState:\n2", messages[1]["content"])
        self.assertTrue(messages[3]["content"].startswith("Reward:\n0\n\nTurn 4:"))
        self.assertIn("Environment state remains the same", messages[3]["content"])
        self.assertIn("Max response length: 100 words (tokens).", messages[3]["content"])

    def test_format_rejection_does_not_advance_either_runner(self):
        responses = iter(["move right", "move right</answer>"])
        episode = run_episode("warmup_right", "base", 0, lambda *_: next(responses), self.protocol, 42)
        first, second = episode["history"]
        self.assertTrue(first["format_error"])
        self.assertIsNone(first["executed_action"])
        self.assertEqual(first["observation"], first["next_observation"])
        self.assertEqual(first["observation"], second["observation"])
        self.assertEqual(first["environment_actions"], 0)
        self.assertEqual(episode["result"], "A")
        self.assertIn("32 actions left", second["messages"][-1]["content"])

    def test_tagged_illegal_action_still_consumes_environment_turn(self):
        responses = iter(["move nowhere</answer>", "move right</answer>"])
        episode = run_episode("warmup_right", "base", 0, lambda *_: next(responses), self.protocol, 42)
        turn = episode["history"][0]
        self.assertFalse(turn["format_error"])
        self.assertFalse(turn["action_is_valid"])
        self.assertEqual(turn["executed_action"], "wait")
        self.assertNotEqual(turn["observation"], turn["next_observation"])

    def test_repeated_format_failures_are_bounded_and_unresolved(self):
        episode = run_episode("warmup_right", "after", 0, lambda *_: "move right", self.protocol, 42)
        self.assertEqual(len(episode["history"]), 32)
        self.assertEqual(episode["result"], "unresolved")
        self.assertEqual(episode["history"][0]["observation"], episode["final_observation"])

    def smoke_episodes(self):
        return [{"policy": p, "layout": layout, "episode": i,
                 "history": [{"format_error": False}], "result": "B"}
                for p in POLICIES for layout in self.protocol["smoke"]["layouts"]
                for i in range(self.protocol["smoke"]["episodes_per_map"])]

    def test_smoke_gates_each_policy_but_never_selects_by_wins(self):
        episodes = self.smoke_episodes()
        self.assertTrue(format_gate(episodes, self.protocol)["passed"])
        episodes[-1]["history"][0]["format_error"] = True
        self.assertFalse(format_gate(episodes, self.protocol)["passed"])
        self.assertFalse(format_gate(episodes[:-1], self.protocol)["complete"])
        duplicate = self.smoke_episodes()
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        self.assertFalse(format_gate(duplicate, self.protocol)["complete"])

    def test_gate_checks_raw_evidence_and_runtime_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            episodes = self.smoke_episodes()
            (root / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
            write_json(root / "manifest.json", {"smoke_only": True, "identity": {"source": "v2"}})
            (root / "COMPLETE").write_text("done")
            gate = format_gate(episodes, self.protocol)
            gate.update(manifest_sha256=file_hash(root / "manifest.json"),
                        episodes_sha256=file_hash(root / "episodes.jsonl"))
            write_json(root / "quality-gate.json", gate)
            verify_smoke(root / "quality-gate.json", {"source": "v2"}, self.protocol)
            with self.assertRaises(ValueError):
                verify_smoke(root / "quality-gate.json", {"source": "changed"}, self.protocol)
            (root / "episodes.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                verify_smoke(root / "quality-gate.json", {"source": "v2"}, self.protocol)


if __name__ == "__main__":
    unittest.main()
