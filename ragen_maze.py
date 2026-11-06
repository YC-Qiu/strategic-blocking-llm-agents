"""RAGEN environment interface for the two existing race fixtures."""

from dataclasses import dataclass

from demo import COSTLY_MAP, SLIDE_MAP
from maze import Maze, Race


@dataclass
class StrategicBlockingConfig:
    layout: str = "slides"
    max_rounds: int = 32


class StrategicBlockingEnv:
    """Use RAGEN's language action path: the manager passes parsed strings."""

    def __init__(self, config):
        layouts = {"slides": SLIDE_MAP, "costly": COSTLY_MAP}
        self.config = config
        self.maze = Maze(layouts[config.layout])
        self.race = None

    def reset(self, seed=None, mode=None):
        # These fixed fixtures have no randomness. Each reset creates a fresh race.
        self.race = Race(self.maze, self.config.max_rounds)
        self.race.advance_b()
        return self.render()

    def step(self, action):
        race = self.race
        if race is None or race.result is not None:
            raise ValueError("step requires an active race")
        requested = str(action).strip().lower()
        valid = requested in race.legal_actions()
        executed = requested if valid else "wait"
        race.act(executed)
        if race.result is None:
            race.advance_b()
        info = dict(success=race.result == "A", result=race.result,
                    action_is_valid=valid, action_is_effective=executed != "wait",
                    requested_action=requested, executed_action=executed,
                    raw_reward=race.reward, a_actions=race.a_actions, b_actions=race.b_actions)
        return self.render(), race.reward, race.result is not None, info

    def render(self, mode=None):
        state = self.race.observe()
        return "\n".join([
            *state["map"],
            f"Round {state['round']}/{state['max_rounds']}; coordinates (row, column), zero-based.",
            f"A {state['a']} -> {state['goal_a']}; B {state['b']} -> {state['goal_b']}.",
            f"Obstacle: {state['block']}; available: {state['block_available']}.",
            "Legal actions: " + ", ".join(state["legal_actions"]),
            "Result: " + (self.race.result or "in progress"),
        ])

    def close(self):
        self.race = None
