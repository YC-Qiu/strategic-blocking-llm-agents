import unittest

from demo import detour, direct, run
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv


class RagenAdapterTests(unittest.TestCase):
    def test_replay_matches_native_race_turn_by_turn(self):
        for layout in ("slides", "costly"):
            for policy in (direct, detour((5, 7), (5, 6))):
                env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
                episode = run("\n".join(env.maze.rows), policy, env.config.max_rounds)
                first = env.reset(seed=123, mode="val")
                for turn in episode["turns"]:
                    action = turn["scripted_action"]
                    if action is None:  # B-only terminal round already processed by step()
                        continue
                    with self.subTest(layout=layout, round=turn["round"]):
                        self.assertEqual(env.race.observe(), turn["observation"])
                        obs, reward, done, info = env.step(action)
                        self.assertEqual(info["executed_action"], action)
                        self.assertEqual(obs, env.render())
                        self.assertEqual(reward, episode["summary"]["reward"] if done else 0)
                self.assertTrue(done)
                self.assertEqual(info["result"], episode["summary"]["result"])
                self.assertEqual(info["success"], info["result"] == "A")
                self.assertEqual(env.reset(seed=123, mode="train"), first)

    def test_invalid_parsed_action_consumes_turn_and_can_timeout(self):
        env = StrategicBlockingEnv(StrategicBlockingConfig(max_rounds=1))
        env.reset()
        _, reward, done, info = env.step("move left")  # wall on the initial map
        self.assertEqual((reward, done, info["result"]), (0, True, "timeout"))
        self.assertFalse(info["action_is_valid"])
        self.assertEqual(info["executed_action"], "wait")
        self.assertEqual(info["a_actions"], 1)
        self.assertEqual(info["b_actions"], 1)
        with self.assertRaises(ValueError):
            env.step("wait")
