"""Check curriculum mechanics and the reward probe's evidence accounting."""
import importlib.util
from pathlib import Path
import unittest

from curriculum import CURRICULUM_MAPS
from demo import direct, run
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("probe_reward_signal", ROOT / "scripts/probe_reward_signal.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def trace(layout, policy):
    env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
    observation = env.reset()
    history = []
    done = False
    while not done:
        action = policy(env.race)
        after, reward, done, info = env.step(action)
        history.append({
            "state": observation, "actions": [action], "reward": reward, "info": info,
            "llm_raw_response": f"<answer>{action}</answer>",
        })
        observation = after
    history.append({"state": observation})
    env.close()
    return history


class CurriculumTests(unittest.TestCase):
    def test_navigation_can_win_and_waiting_can_lose(self):
        for layout in ("warmup_right", "warmup_left", "warmup_two_right", "warmup_two_left"):
            with self.subTest(layout=layout):
                direct_case = run(CURRICULUM_MAPS[layout], direct)
                wait_case = run(CURRICULUM_MAPS[layout], lambda race: "wait")
                self.assertEqual(direct_case["summary"]["result"], "A")
                self.assertEqual(wait_case["summary"]["result"], "B")
                self.assertEqual(direct_case["summary"]["a_actual"], 2 if "two" in layout else 1)

    def test_short_block_changes_loss_to_win(self):
        layout = CURRICULUM_MAPS["blocking_short"]
        direct_case = run(layout, direct)
        block_case = run(layout, lambda race: "block up" if race.block is None else direct(race))
        self.assertEqual(direct_case["summary"]["result"], "B")
        self.assertEqual(block_case["summary"]["result"], "A")
        self.assertEqual(block_case["summary"]["a_actual"], 5)
        self.assertEqual(block_case["summary"]["b_full"], 12)

    def test_seed_does_not_create_new_layout(self):
        for layout in CURRICULUM_MAPS:
            env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
            self.assertEqual(env.reset(seed=1), env.reset(seed=2))

    def test_adapter_replay_matches_native_for_new_maps(self):
        for layout in CURRICULUM_MAPS:
            record = run(CURRICULUM_MAPS[layout], direct, max_rounds=32)
            env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
            env.reset()
            for step in record["turns"]:
                if step["scripted_action"] is None:
                    continue
                self.assertEqual(env.race.observe(), step["observation"])
                _, _, done, info = env.step(step["scripted_action"])
            self.assertTrue(done)
            self.assertEqual(info["result"], record["summary"]["result"])


class ProbeTests(unittest.TestCase):
    def test_mixed_terminal_returns_are_informative(self):
        histories = [trace("warmup_right", direct), trace("warmup_right", lambda race: "wait")]
        summary = probe.summarize_histories(histories, ["warmup_right"], 2)
        self.assertEqual(summary["groups"][0]["returns"], [1.0, -1.0])
        self.assertEqual(summary["groups"][0]["reward_variance"], 1.0)
        self.assertEqual(summary["informative_group_count"], 1)
        self.assertFalse(summary["training_grpo_group_alignment_verified"])

    def test_all_successes_are_not_informative(self):
        histories = [trace("warmup_right", direct)] * 2
        self.assertEqual(probe.summarize_histories(histories, ["warmup_right"], 2)["informative_group_count"], 0)

    def test_truncation_is_not_a_loss_or_valid_terminal_group(self):
        completed = trace("warmup_right", direct)
        truncated = [{"state": completed[0]["state"], "actions": [],
                      "llm_raw_response": "I do not know.", "reward": 0}]
        summary = probe.summarize_histories([completed, truncated], ["warmup_right"], 2)
        self.assertEqual(summary["episodes"][1]["result"], "unresolved")
        self.assertEqual(summary["episodes"][1]["format_error_count"], 1)
        self.assertEqual(summary["episodes"][1]["known_action_validity_count"], 0)
        self.assertEqual(summary["informative_group_count"], 0)

    def test_wrong_episode_count_fails(self):
        with self.assertRaisesRegex(ValueError, "expected 4"):
            probe.summarize_histories([trace("warmup_right", direct)], ["warmup_right"], 4)

    def test_unknown_initial_state_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown initial"):
            probe.summarize_histories([trace("warmup_left", direct)] * 2, ["warmup_right"], 2)

    def test_rewards_must_match_terminal_outcome(self):
        history = trace("warmup_right", direct)
        history[0]["reward"] = -1
        with self.assertRaisesRegex(ValueError, "disagree"):
            probe.episode_summary(history)

    def test_sampled_config_uses_same_protocol_and_expected_budget(self):
        config = probe.make_config(ROOT, Path("/tmp/probe-output"), probe.DEFAULT_LAYOUTS, 4, 42, 1.0)
        validation = config["es_manager"]["val"]
        self.assertEqual(validation["env_groups"] * validation["group_size"], 28)
        self.assertTrue(config["actor_rollout_ref"]["rollout"]["val_kwargs"]["do_sample"])
        self.assertFalse(config["agent_proxy"]["enable_think"])
        self.assertEqual(config["es_manager"]["format_penalty"], 0.0)
        self.assertNotIn("total_training_steps", config["trainer"])
        for index, layout in enumerate(probe.DEFAULT_LAYOUTS):
            self.assertEqual(config["custom_envs"][f"Probe{index}"]["env_config"]["layout"], layout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
