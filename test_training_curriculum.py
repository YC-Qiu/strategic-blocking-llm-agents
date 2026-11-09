"""Validate the blocking curriculum before allocating GPU resources."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).parent


def build_config(mode, adapter="/verified/source/adapter"):
    launcher = (ROOT / "scripts/run_ragen.sh").read_text(encoding="utf-8")
    source = launcher.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory)
        config_file = output / "generated.yaml"
        env = {"PROJECT": str(ROOT), "PILOT_ADAPTER": adapter}
        inspector = SimpleNamespace(inspect_adapter=lambda path: {
            "adapter_path": str(path), "tensor_count": 336,
            "tensor_sha256": "a" * 64, "optimizer_restored": False,
        })
        with patch.dict(os.environ, env), patch.object(sys, "path", list(sys.path)), \
             patch.object(sys, "argv", ["config", str(config_file), str(output), mode]), \
             patch.dict(sys.modules, {"verify_warm_start": inspector}):
            exec(compile(source, "run_ragen_config", "exec"), {})
        manifest = output / "warm-start.json"
        return yaml.safe_load(config_file.read_text()), manifest.exists()


class BlockingCurriculumTests(unittest.TestCase):
    def test_each_reward_group_has_one_layout_and_eight_samples(self):
        config, recorded = build_config("train-blocking")
        train = config["es_manager"]["train"]
        self.assertTrue(recorded)
        self.assertEqual(train["group_size"], 8)
        self.assertEqual(train["env_groups"], 3)
        self.assertEqual(train["env_configs"]["n_groups"], [1, 1, 1])
        layouts = [config["custom_envs"][tag]["env_config"]["layout"]
                   for tag in train["env_configs"]["tags"]]
        self.assertEqual(layouts, ["blocking_short", "warmup_two_right", "warmup_two_left"])

    def test_retention_validation_covers_both_navigation_difficulties(self):
        config, _ = build_config("train-blocking")
        val = config["es_manager"]["val"]
        self.assertEqual(val["group_size"], 1)
        self.assertEqual(val["env_groups"], 5)
        self.assertEqual(sum(val["env_configs"]["n_groups"]), 5)
        layouts = {config["custom_envs"][tag]["env_config"]["layout"]
                   for tag in val["env_configs"]["tags"]}
        self.assertEqual(layouts, {"blocking_short", "warmup_two_right", "warmup_two_left",
                                  "warmup_right", "warmup_left"})

    def test_warm_start_and_training_controls_match_two_step_stage(self):
        old, _ = build_config("train-two-step")
        new, _ = build_config("train-blocking")
        self.assertEqual(old["actor_rollout_ref"], new["actor_rollout_ref"])
        self.assertEqual(new["actor_rollout_ref"]["model"]["lora_adapter_path"], str(Path("/verified/source/adapter")))
        self.assertEqual(new["algorithm"], old["algorithm"])
        self.assertEqual(new["trainer"]["total_training_steps"], 5)
        self.assertEqual(new["trainer"]["save_freq"], 5)
        self.assertEqual(new["es_manager"]["format_penalty"], 0.0)
        instructions = {new["custom_envs"][tag]["env_instruction"]
                        for tag in new["es_manager"]["train"]["env_configs"]["tags"]}
        self.assertEqual(len(instructions), 1)

    def test_missing_source_adapter_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "PILOT_ADAPTER"):
            build_config("train-blocking", adapter="")


if __name__ == "__main__":
    unittest.main()
