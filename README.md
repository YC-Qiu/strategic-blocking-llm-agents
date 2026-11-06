# Taking a Detour to Win — Strategic Blocking in LLM Agents

YC Qiu and Prakeerth Prasad · Agentic AI · Milestone 1

Run inside this directory with Python 3.10+ (standard library only):

```powershell
python demo.py
python demo.py --trace episodes.json
python -m unittest -v
```

## Reproductions

The 11-row, 14-column main map was transcribed from the tiles on slides 3–4 of the [project presentation](https://docs.google.com/presentation/d/10VW3zM-tbud9UVQoxef7KA3oxMRh8gwywWqiOLcqfa0/edit). Coordinates count from zero, in (row, column) order.

| Map and A policy | Winner | Actual A / B actions | Full A / B actions |
| --- | --- | --- | --- |
| Slides 3–4, direct | B | 8 / 9 | 13 / 9 |
| Slides 3–4, block on arrival | A | 22 / 22 | 22 / 25 |
| Slide 7, block now | A | 6 / 6 | 6 / 12 |
| Slide 7, wait three turns | A | 9 / 9 | 9 / 18 |
| Modified map, direct | A | 6 / 6 | 6 / 9 |
| Modified map, block on arrival | B | 12 / 13 | 15 / 13 |

Actual counts stop at the first arrival. Full counts add the shortest remaining distance from the terminal state, assuming direct movement thereafter.

- **Main race.** The direct shortest paths are disjoint. A moves right twice and down twice, from (3,5) to (5,7). On round 5 B reaches (5,5) and A blocks (5,6). B has already spent 5 actions and now has 20 left. A takes 4 steps back and follows its original 13-step route. At round 22 A wins with B still 3 steps away.
- **Timing experiment.** Slide 7 uses its own map and omits A's goal. Both runs use the same added A goal at the upper-left open cell. B starts at the fork (1,0). At placement it is either 1 or 4 steps inside the corridor. Three extra steps in and three back out explain the six-step difference. Both policies win with this choice of A goal. The slide supports a comparison of travel costs.
- **Costly-detour control.** This authored variant moves A's goal to (1,9) and opens (6,3). A can now win in 6 actions by moving directly. Blocking still delays B from 9 to 13 total actions, but A would need 15.

## Environment and traces

`maze.py` contains the generic rules. B moves first each round using BFS from its current position, breaking shortest-path ties in up, right, down, left order. Runners can share or cross a tile. Only the permanent obstacle affects movement. A may place it once, on an orthogonally adjacent open cell, excluding either goal and B's current cell. Both runners must still have a path from their current positions to their respective goals. An invalid action raises an error before consuming A's turn.

At an A decision point call `race.observe()` after `race.advance_b()`, then apply one legal string with `race.act(action)`. If B reaches its goal, the round stops before A acts. The default cap is 100 complete rounds and arrival on the last round takes precedence over timeout. Terminal reward from A's perspective is +1 for an A win, −1 for a B win, and 0 for timeout. Intermediate rewards are zero.

`episodes.json` contains all six executed traces. Each round records B's move, A's full observation after B moves, the scripted action, the resulting map, the outcome and the reward. The map uses `#` for walls, `.` for open cells, `A/B` for runners, `a/b` for goals, `X` for the obstacle and `*` for shared occupancy. Goal coordinates are always available even when a runner covers the goal marker. The final B-only turn has no A decision or action.

The policies in `demo.py` are deterministic baselines. LLM inference, RAGEN training, loss/reward curves, and evaluation on held-out mazes are subsequent milestones.
