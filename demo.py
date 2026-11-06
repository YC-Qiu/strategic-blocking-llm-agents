"""Run the slide reproductions with scripted A policies and a BFS opponent."""

import argparse
import json
from pathlib import Path

from maze import DIRECTIONS, Maze, Race, adjacent


# Slides 3 and 4. Each character corresponds to one slide tile.
SLIDE_MAP = """##############
#####.....#.a#
#####.###.#.##
#.B.#A..#...##
#.#.###.######
#.#.....######
#.####.#######
#.#....b######
#.#.##########
#...##########
##############"""

# Slide 7 shows B after moving and omits GA. Start B at the fork and mark
# the upper-left floor cell as GA for both runs; only B's route is compared.
TIMING_MAP = """a....A#
B.....b
.#####.
......."""

# Control derived from the main map: move GA closer and open (6, 3),
# shortening B's fallback route without changing B's direct route.
COSTLY_MAP = SLIDE_MAP.replace("#####.....#.a#", "#####....a#..#").replace(
    "#.####.#######", "#.#.##.#######")


def direction_to(start, target):
    return next(name for name, delta in DIRECTIONS.items() if adjacent(start, delta) == target)


def move_toward(race, destination):
    route = race.maze.path(race.a, destination, race.block)
    return "move " + direction_to(race.a, route[1])


def direct(race):
    return move_toward(race, race.ga)


def detour(stand, target, wait_turns=0):
    """Create a per-episode policy with a fixed placement site."""
    waits_left = wait_turns

    def policy(race):
        nonlocal waits_left
        if race.block is not None:
            return direct(race)
        if race.a != stand:
            return move_toward(race, stand)
        if waits_left > 0:
            waits_left -= 1
            return "wait"
        return "block " + direction_to(stand, target)

    return policy


def run(layout, policy, max_rounds=100):
    race = Race(Maze(layout), max_rounds)
    initial = race.observe()
    turns = []
    while race.result is None:
        before_b = race.b
        race.advance_b()
        decision = race.observe()
        action = None
        if race.result is None:
            action = policy(race)
            race.act(action)
        turns.append(dict(round=race.b_actions, b_from=before_b, b_to=race.b,
                          observation=decision if action is not None else None,
                          scripted_action=action, after=race.observe(),
                          result=race.result, reward=race.reward))
    # Stop at the first finisher. Measure remaining distances on the final map
    # separately without fabricating subsequent turns for the losing runner.
    remaining_a = len(race.maze.path(race.a, race.ga, race.block)) - 1
    remaining_b = len(race.maze.path(race.b, race.gb, race.block)) - 1
    summary = dict(result=race.result, reward=race.reward, a_actual=race.a_actions,
                   b_actual=race.b_actions, a_full=race.a_actions + remaining_a,
                   b_full=race.b_actions + remaining_b)
    return dict(initial=initial, turns=turns, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="write full episode records as JSON")
    args = parser.parse_args()
    stand, target = (5, 7), (5, 6)
    cases = [
        ("slides/direct", SLIDE_MAP, direct),
        ("slides/block", SLIDE_MAP, detour(stand, target)),
        ("timing/immediate", TIMING_MAP, detour((0, 5), (1, 5))),
        ("timing/wait-3", TIMING_MAP, detour((0, 5), (1, 5), 3)),
        ("control/direct", COSTLY_MAP, direct),
        ("control/block", COSTLY_MAP, detour(stand, target)),
    ]
    episodes = {}
    print("case                  winner reward  actual A/B   full A/B")
    for name, layout, policy in cases:
        episode = run(layout, policy)
        episodes[name] = episode
        s = episode["summary"]
        print(f"{name:22}{s['result']:>5}{s['reward']:>7}  "
              f"{s['a_actual']:>4}/{s['b_actual']:<4}   {s['a_full']:>3}/{s['b_full']}")
    if args.trace:
        args.trace.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
