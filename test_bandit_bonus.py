"""CPU regression checks for isolated Bandit setup and evaluation accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
import bandit_bonus as bonus
from verify_strategic_environment import audit


class BanditBonusTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "bandit_bonus.yaml").read_text())

    def test_independent_training_configuration(self):
        self.assertEqual(self.cfg["model_path"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertNotIn("lora_adapter_path", self.cfg["actor_rollout_ref"]["model"])
        self.assertEqual(self.cfg["trainer"]["resume_mode"], "disable")
        self.assertEqual(self.cfg["trainer"]["total_training_steps"], 5)
        self.assertEqual(self.cfg["algorithm"]["adv_estimator"], "grpo")
        self.assertEqual(self.cfg["agent_proxy"]["max_turn"], 1)
        self.assertEqual(self.cfg["agent_proxy"]["reward_normalization"]["method"], "identity")
        train = self.cfg["es_manager"]["train"]
        self.assertEqual(train["env_groups"] * train["group_size"], 16)
        for mode in ("train", "val"):
            for tag in self.cfg["es_manager"][mode]["env_configs"]["tags"]:
                self.assertEqual(self.cfg["custom_envs"][tag]["env_config"]["split"], "train")

    def test_evaluation_uses_test_names_and_shared_base_without_mutating_training(self):
        self.cfg["actor_rollout_ref"]["model"]["lora_adapter_path"] = "unwanted"
        before = deepcopy(self.cfg)
        actual = bonus.make_evaluation_config(self.cfg, "/cached/base", count=200)
        self.assertEqual(self.cfg, before)
        self.assertNotIn("lora_adapter_path", actual["actor_rollout_ref"]["model"])
        self.assertEqual(actual["model_path"], actual["actor_rollout_ref"]["model"]["path"])
        self.assertEqual(actual["es_manager"]["val"]["env_configs"]["tags"], ["BonusBanditTest"])
        self.assertEqual(actual["custom_envs"]["BonusBanditTest"]["env_config"]["split"], "test")
        self.assertEqual(actual["es_manager"]["val"]["group_size"], 1)
        self.assertEqual(actual["es_manager"]["val"]["env_groups"], 200)

    def test_reward_math_separates_luck_from_choice_quality(self):
        self.assertAlmostEqual(bonus.expected_reward("low"), 0.1)
        self.assertAlmostEqual(bonus.expected_reward("high"), 0.25)
        self.assertAlmostEqual((bonus.expected_reward("low") + bonus.expected_reward("high"))/2, 0.175)
        rows = [{"choice": "high", "reward": 0, "expected_reward": 0.25},
                {"choice": "low", "reward": 0.1, "expected_reward": 0.1},
                {"choice": "invalid", "reward": 0, "expected_reward": 0}]
        result = bonus.metrics(rows)
        self.assertEqual(result["high_arm_rate"], 1/3)
        self.assertEqual(result["invalid"], 1)
        self.assertAlmostEqual(result["mean_observed_reward"], 0.1/3)
        self.assertAlmostEqual(result["mean_expected_reward"], 0.35/3)
        with self.assertRaises(ValueError):
            bonus.metrics([])

    def test_custom_environment_audit_checks_beneficial_and_costly_detours(self):
        result = audit()
        self.assertFalse(result["registration"]["checked"])
        self.assertEqual([row["result"] for row in result["scripted_cases"]], ["B", "A", "A", "B"])
        self.assertTrue(result["invalid_parsed_action_consumes_turn"])

    def test_comparison_rejects_unmatched_initial_states_before_scoring(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            record = {"episode": 0, "reset_seed": 1, "internal_seed": 0,
                      "low_name": "low", "high_name": "high", "observation_sha256": "same"}
            for name in ("base", "adapter"):
                (run / name).mkdir()
                (run / name / "episodes.jsonl").write_text(json.dumps(record) + "\n")
                record["observation_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "not matched"):
                bonus.report(SimpleNamespace(run=run))

    def test_report_uses_real_score_metric_and_requires_complete_evidence(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg = deepcopy(self.cfg)
            cfg["trainer"]["total_training_steps"] = 1
            (run / "composed.yaml").write_text(yaml.safe_dump(cfg))
            (run / "run.log").write_text(
                "step:1 - actor/pg_loss:0.01 - critic/score/mean:0.25 - "
                "train/charts/avg_episode_return:0.0 - train/BonusBanditTrain/success:1.0 - "
                "critic/advantages/min:-1.0 - critic/advantages/max:1.0\n")
            row = {"episode": 0, "reset_seed": 1, "internal_seed": 0,
                   "low_name": "low", "high_name": "high", "observation_sha256": "same",
                   "choice": "high", "reward": 0, "expected_reward": 0.25}
            controls = [dict(row, policy=p) for p in ("uniform_random", "first_position", "always_low", "high_oracle")]
            trace = dict(env_id=0, turn=0, seed=1, seed_scheme="sha256-v2", prompt_sha256="same")
            for name in ("base", "adapter"):
                (run / name).mkdir()
                (run / name / "episodes.jsonl").write_text(json.dumps(row) + "\n")
                (run / name / "controls.json").write_text(json.dumps(controls))
                (run / name / "generation-trace.jsonl").write_text(json.dumps(trace) + "\n")
            plotting = MagicMock()
            figure, axes = MagicMock(), [MagicMock() for _ in range(3)]
            plotting.pyplot.subplots.return_value = (figure, axes)
            with patch.object(bonus, "EVAL_COUNT", 1), patch.dict(sys.modules, {
                    "matplotlib": plotting, "matplotlib.pyplot": plotting.pyplot}):
                bonus.report(SimpleNamespace(run=run))
            result = json.loads((run / "comparison.json").read_text())
            self.assertEqual(result["updates_with_nonzero_advantages"], 1)
            self.assertEqual(result["results"]["adapter"]["mean_observed_reward"], 0)
            self.assertEqual(result["results"]["adapter"]["mean_expected_reward"], 0.25)
            self.assertEqual(axes[1].plot.call_args.args[1], (0.25,))
            self.assertEqual(figure.savefig.call_count, 2)
            self.assertTrue((run / "COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
