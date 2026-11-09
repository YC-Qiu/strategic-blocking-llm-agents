"""Small training-only fixtures for diagnosing sparse terminal rewards.

These hand-authored maps preserve the original rules. They are curriculum
candidates, not held-out evaluation data or evidence of model improvement.
"""

ONE_STEP_RIGHT = """#######
#Aa...#
#.###.#
#B..b.#
#######"""

TWO_STEP_RIGHT = """#######
#A.a..#
#.###.#
#B...b#
#######"""

SHORT_BLOCK = """########
#B...b.#
#.#A##.#
#.....a#
########"""


def mirror(layout):
    return "\n".join(row[::-1] for row in layout.splitlines())


CURRICULUM_MAPS = {
    "warmup_right": ONE_STEP_RIGHT,
    "warmup_left": mirror(ONE_STEP_RIGHT),
    "warmup_two_right": TWO_STEP_RIGHT,
    "warmup_two_left": mirror(TWO_STEP_RIGHT),
    "blocking_short": SHORT_BLOCK,
}
