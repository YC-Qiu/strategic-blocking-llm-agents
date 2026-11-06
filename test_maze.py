import unittest

from demo import COSTLY_MAP, SLIDE_MAP, TIMING_MAP, detour, direct, run
from maze import Maze, Race


class Reproductions(unittest.TestCase):
    def test_direct(self):
        maze = Maze(SLIDE_MAP)
        a, b = [maze.path(maze.markers[k], maze.markers[k.lower()]) for k in "AB"]
        self.assertTrue(set(a).isdisjoint(b))
        episode = run(SLIDE_MAP, direct)
        self.assertEqual(episode["summary"], dict(result="B", reward=-1, a_actual=8,
                         b_actual=9, a_full=13, b_full=9))
        self.assertIsNone(episode["turns"][-1]["scripted_action"])

    def test_detour(self):
        episode = run(SLIDE_MAP, detour((5, 7), (5, 6)))
        self.assertEqual(episode["summary"], dict(result="A", reward=1, a_actual=22,
                         b_actual=22, a_full=22, b_full=25))
        turns = episode["turns"]
        self.assertEqual(turns[4]["scripted_action"], "block left")
        self.assertEqual(turns[4]["b_to"], (5, 5))
        self.assertEqual(turns[5]["b_to"], (5, 4))
        self.assertEqual(turns[8]["after"]["a"], episode["initial"]["a"])
        for turn in turns[4:]:
            self.assertEqual(turn["after"]["block"], (5, 6))
            self.assertNotIn((5, 6), (turn["after"]["a"], turn["after"]["b"]))

    def test_block_timing_from_same_start(self):
        for waits, depth, full in [(0, 1, 12), (3, 4, 18)]:
            with self.subTest(waits=waits):
                episode = run(TIMING_MAP, detour((0, 5), (1, 5), waits))
                turn = episode["turns"][waits]
                self.assertEqual(turn["b_to"], (1, depth))
                self.assertEqual(turn["scripted_action"], "block down")
                self.assertEqual(episode["summary"]["b_full"], full)

    def test_costly_detour_flips_a_win_to_loss(self):
        self.assertEqual(run(COSTLY_MAP, direct)["summary"]["result"], "A")
        self.assertEqual(run(COSTLY_MAP, detour((5, 7), (5, 6)))["summary"]["result"], "B")


class Rules(unittest.TestCase):
    def test_first_arrival_and_timeout(self):
        # Both are one step away. B reaches its goal before A can act.
        tie = run("Aa\nBb", direct, max_rounds=1)
        self.assertEqual(tie["summary"]["result"], "B")
        self.assertEqual(tie["summary"]["a_actual"], 0)
        timeout = run(SLIDE_MAP, lambda race: "wait", max_rounds=1)
        self.assertEqual(timeout["summary"]["result"], "timeout")
        self.assertEqual(timeout["summary"]["reward"], 0)
        self.assertEqual(run("Aa#\nB.b", direct, 1)["summary"]["result"], "A")

    def test_cannot_cut_either_route(self):
        race = Race(Maze("A.a\n###\nB.b"))
        race.advance_b()
        self.assertFalse(race.can_block((0, 1)))  # Cuts A off.
        race = Race(Maze("a.A##\nB...b"))
        race.advance_b()
        self.assertFalse(race.can_block((1, 2)))  # Cuts B's only route to b.

    def test_occupied_cell_and_goals_cannot_be_blocked(self):
        race = Race(Maze("aA.b\nB..."))
        race.advance_b()
        self.assertEqual(race.b, (0, 0))
        # Move A beside GB and let B take its next step.
        race.act("move right")
        race.advance_b()
        self.assertEqual(race.b, (0, 1))
        self.assertFalse(race.can_block(race.b))
        self.assertFalse(race.can_block(race.gb))

    def test_illegal_actions_are_atomic(self):
        race = Race(Maze(TIMING_MAP))
        with self.assertRaises(ValueError):
            race.act("wait")
        race.advance_b()
        initial = race.observe()
        for action in ("move right", "block up", "block right", "teleport"):
            with self.assertRaises(ValueError):
                race.act(action)
            self.assertEqual(race.observe(), initial)
        race.act("block down")
        with self.assertRaises(ValueError):
            race.act("wait")
        race.advance_b()
        self.assertNotIn("move down", race.legal_actions())
        self.assertFalse(any(a.startswith("block") for a in race.legal_actions()))


if __name__ == "__main__":
    unittest.main()
