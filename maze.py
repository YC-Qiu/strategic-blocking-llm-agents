"""Deterministic two-runner race. Coordinates are zero-based (row, column)."""

from collections import deque

Cell = tuple[int, int]
DIRECTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def adjacent(cell: Cell, delta: Cell) -> Cell:
    return cell[0] + delta[0], cell[1] + delta[1]


class Maze:
    def __init__(self, layout: str):
        self.rows = tuple(layout.strip().splitlines())
        if not self.rows or len({len(row) for row in self.rows}) != 1:
            raise ValueError("Expected a rectangular map")
        self.floor = set()
        self.markers = {}
        for r, row in enumerate(self.rows):
            for c, tile in enumerate(row):
                if tile not in "#.ABab":
                    raise ValueError(f"Unknown map tile {tile!r}")
                if tile != "#":
                    self.floor.add((r, c))
                if tile in "ABab":
                    if tile in self.markers:
                        raise ValueError(f"Duplicate marker {tile}")
                    self.markers[tile] = (r, c)
        if self.markers.keys() != set("ABab"):
            raise ValueError("Expected starts A/B and goals a/b")
        for runner in "AB":
            if self.path(self.markers[runner], self.markers[runner.lower()]) is None:
                raise ValueError(f"{runner} cannot reach its goal")

    def path(self, start: Cell, goal: Cell, block: Cell | None = None):
        """BFS returns a shortest path including endpoints, or None."""
        if start not in self.floor or goal not in self.floor or block in (start, goal):
            return None
        frontier = deque([start])
        parent = {start: None}
        while frontier:
            cell = frontier.popleft()
            if cell == goal:
                route = []
                while cell is not None:
                    route.append(cell)
                    cell = parent[cell]
                return route[::-1]
            for delta in DIRECTIONS.values():
                neighbor = adjacent(cell, delta)
                if neighbor in self.floor and neighbor != block and neighbor not in parent:
                    parent[neighbor] = cell
                    frontier.append(neighbor)
        return None


class Race:
    """Starts on B's turn. advance_b() precedes each observation and A action."""

    def __init__(self, maze: Maze, max_rounds: int = 100):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.maze = maze
        self.a, self.b = (maze.markers[k] for k in "AB")
        self.ga, self.gb = (maze.markers[k] for k in "ab")
        self.block = None
        self.a_actions = 0
        self.b_actions = 0
        self.max_rounds = max_rounds
        self.turn = "B"
        self.result = None

    @property
    def reward(self):
        return {"A": 1, "B": -1}.get(self.result, 0)

    def advance_b(self):
        if self.turn != "B" or self.result is not None:
            raise ValueError("B cannot act now")
        self.b = self.maze.path(self.b, self.gb, self.block)[1]
        self.b_actions += 1
        if self.b == self.gb:
            self.result = "B"
        self.turn = None if self.result else "A"

    def can_block(self, target: Cell):
        if self.block is not None or target not in self.maze.floor:
            return False
        if target in (self.b, self.ga, self.gb):
            return False
        if abs(target[0] - self.a[0]) + abs(target[1] - self.a[1]) != 1:
            return False
        return all(self.maze.path(start, goal, target) is not None
                   for start, goal in ((self.a, self.ga), (self.b, self.gb)))

    def legal_actions(self):
        if self.turn != "A" or self.result is not None:
            return []
        actions = ["wait"]
        for direction, delta in DIRECTIONS.items():
            target = adjacent(self.a, delta)
            if target in self.maze.floor and target != self.block:
                actions.append(f"move {direction}")
            if self.can_block(target):
                actions.append(f"block {direction}")
        return actions

    def act(self, action: str):
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r}")
        if action != "wait":
            verb, direction = action.split()
            target = adjacent(self.a, DIRECTIONS[direction])
            if verb == "move":
                self.a = target
            else:
                self.block = target
        self.a_actions += 1
        if self.a == self.ga:
            self.result = "A"
        elif self.b_actions == self.max_rounds:
            self.result = "timeout"
        self.turn = None if self.result else "B"

    def observe(self):
        grid = [list(row.replace("A", ".").replace("B", ".")) for row in self.maze.rows]
        for cell, symbol in ((self.block, "X"), (self.a, "A"), (self.b, "B")):
            if cell is not None:
                grid[cell[0]][cell[1]] = symbol
        if self.a == self.b:
            grid[self.a[0]][self.a[1]] = "*"
        return dict(map=["".join(row) for row in grid], a=self.a, b=self.b,
                    goal_a=self.ga, goal_b=self.gb, block=self.block,
                    block_available=self.block is None, round=self.b_actions,
                    max_rounds=self.max_rounds, turn=self.turn,
                    legal_actions=self.legal_actions())
