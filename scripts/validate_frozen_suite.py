"""Validate frozen evaluation fixtures with deterministic policies, not an LLM."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maze import Maze, Race, DIRECTIONS
from frozen_eval import load_suite


def direct_policy(layout):
    race = Race(Maze(layout), 32)
    actions = []
    while race.result is None:
        race.advance_b()
        if race.result is not None:
            break
        route = race.maze.path(race.a, race.ga, race.block)
        delta = (route[1][0] - race.a[0], route[1][1] - race.a[1])
        action = "move " + next(name for name, value in DIRECTIONS.items() if value == delta)
        actions.append(action)
        race.act(action)
    return {"result": race.result, "actions": actions}


def winning_policy(layout):
    """Find a shortest winning A action sequence by exhaustive state search."""
    initial = Race(Maze(layout), 32)
    initial.advance_b()
    frontier = deque([(initial, [])])
    seen = set()
    while frontier:
        race, actions = frontier.popleft()
        if race.result == "A":
            return actions
        if race.result is not None:
            continue
        key = (race.a, race.b, race.block)
        if key in seen:
            continue
        seen.add(key)
        for action in race.legal_actions():
            child = deepcopy(race)
            child.act(action)
            if child.result is None:
                child.advance_b()
            frontier.append((child, actions + [action]))
    raise ValueError("No winning policy within 32 rounds.")


def validate():
    suite = load_suite()
    output = []
    for case in suite["maps"]:
        direct = direct_policy(case["layout"])
        oracle = winning_policy(case["layout"])
        if direct["result"] != case["expected_direct_outcome"]:
            raise ValueError(f"Unexpected direct-policy outcome: {case['name']}")
        output.append({"name": case["name"], "direct": direct, "winning_actions": oracle})
    return output


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
