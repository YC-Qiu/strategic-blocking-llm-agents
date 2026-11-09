"""Check fixture separation and the portable evaluation's action protocol."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
from curriculum import CURRICULUM_MAPS
from frozen_eval import FROZEN_MAPS, load_suite
from validate_frozen_suite import validate
from evaluate_colab_transfer import messages_for, parse_action, request_seed, run_episode


class FrozenTransferTests(unittest.TestCase):
    def test_fixture_separation_and_rule_solvability(self):
        self.assertEqual(len(FROZEN_MAPS), 8)
        self.assertEqual(len(set(FROZEN_MAPS.values())), 8)
        self.assertFalse(set(FROZEN_MAPS.values()) & set(CURRICULUM_MAPS.values()))
        checks = validate()
        self.assertEqual(sum(row["direct"]["result"] == "A" for row in checks), 4)

    def test_no_new_fixture_selected_by_training_modes(self):
        from test_training_curriculum import build_config
        for mode in ("train-one", "train-warmup", "train-five", "train-sync", "train-balanced", "train-two-step", "train-blocking"):
            cfg, _ = build_config(mode)
            tags = cfg["es_manager"]["train"]["env_configs"]["tags"]
            self.assertFalse({cfg["custom_envs"][t]["env_config"]["layout"] for t in tags} & set(FROZEN_MAPS))

    def test_format_rejection_is_distinct_from_an_illegal_action(self):
        self.assertEqual(parse_action("<answer>move right</answer>"), ("move right", False))
        self.assertEqual(parse_action("<answer>move nowhere</answer>"), ("move nowhere", False))
        self.assertTrue(parse_action("move right")[1])
        self.assertTrue(parse_action("<answer>wait</answer><answer>wait</answer>")[1])

    def test_history_is_bounded_and_keeps_the_current_observation(self):
        history = [dict(observation=str(i), response="response") for i in range(4)]
        messages = messages_for("current", history, dict(instruction="rules", history_turns=2))
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[1]["content"], "2")
        self.assertEqual(messages[-1]["content"], "current")

    def test_portable_runner_executes_actions_and_records_invalid_outputs(self):
        responses = iter(["bad format", "<answer>move right</answer>"])
        episode = run_episode("warmup_right", "base", 0, lambda *_: next(responses),
                              dict(instruction="rules", history_turns=2), 20261112)
        self.assertEqual(episode["result"], "A")
        self.assertEqual(episode["history"][0]["executed_action"], "wait")
        self.assertTrue(episode["history"][0]["format_error"])

    def test_colab_seed_domain_separates_maps_episodes_and_turns(self):
        seeds = {request_seed(20261112, name, episode, turn)
                 for name in FROZEN_MAPS for episode in range(50) for turn in range(32)}
        self.assertEqual(len(seeds), 8 * 50 * 32)
        self.assertTrue(all(0 <= value < 2**63 for value in seeds))


if __name__ == "__main__":
    unittest.main()
