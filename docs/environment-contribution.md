# New environment contribution: Strategic Blocking for RAGEN

## Contribution and boundary

We added a custom text-based two-runner maze environment to an existing LLM RL
framework, RAGEN. The contribution includes a simulator, deterministic opponent,
language-action adapter, environment configurations, scripted controls, rule
tests and a working LoRA/GRPO training/evaluation workflow. It does not claim to
introduce GRPO, LoRA or the RAGEN framework itself.

This is an extension in our project and local RAGEN checkout. Upstream acceptance
or an official RAGEN merge is not claimed. The course rubric requests adding an
environment to an existing repository; this document supplies the implementation
and integration evidence for that criterion, subject to instructor assessment.

## Implementation map

| Component | Repository location | Responsibility |
|---|---|---|
| Simulator | `maze.py` | Map rules, legal actions, turn order, blocking and termination |
| Scripted controls | `demo.py`, `curriculum.py` | Reproductions, navigation fixtures and detour controls |
| RAGEN adapter | `ragen_maze.py` | Config dataclass; reset, step, render and close |
| Registration contract | `RAGEN.md` | Environment and configuration registry entries |
| Configuration | `strategic_blocking.yaml` | Environment tags, instructions, manager budgets |
| Training integration | `scripts/run_ragen.sh` | Reduced-memory LoRA/GRPO launcher and curriculum stages |
| Rule tests | `test_maze.py`, `test_ragen_maze.py`, `test_curriculum.py` | Simulator and adapter contract checks |
| Integration audit | `scripts/verify_strategic_environment.py` | Registry identity and native/adapter replay equivalence |

## Environment contract

The agent controls runner A; B follows a fixed shortest-path BFS opponent with
deterministic tie-breaking. B moves first each round. Reset advances B before
showing the first decision state. Each adapter step applies one A action and
then advances B unless the episode has ended.

A may move one cell, wait, or spend its single permanent obstacle on a legal
adjacent cell while preserving reachability of both goals. The observation
contains the grid, positions/goals, legal actions, round limit and obstacle state.
The only task rewards are A win +1, B win -1 and timeout 0. No intermediate
shaping reward is added. The fixed opponent is not a co-trained agent.

The interface returns `(observation, reward, done, info)` from step. Info includes
success, result, requested/executed action, action validity, raw reward and action
counts. A parsed illegal action becomes wait and consumes an environment turn.
A malformed output can be rejected by RAGEN before step is called; such a manager
request does not advance either runner. Those cases must be reported separately.

## Reproduce registration and verification

Follow `RAGEN.md` and `docs/local-training.md` for the existing installation and
compatibility patches. With the project on PYTHONPATH, the RAGEN registry requires:

```python
from ragen_maze import StrategicBlockingEnv, StrategicBlockingConfig
REGISTERED_ENVS["strategic_blocking"] = StrategicBlockingEnv
REGISTERED_ENV_CONFIGS["strategic_blocking"] = StrategicBlockingConfig
```

Do not append duplicate registry entries if they are already installed. Then run:

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -q
python scripts/verify_strategic_environment.py \
  --ragen-dir "$HOME/RAGEN" \
  --output results/environment-contribution.json
```

The audit fails if the live registry points to different classes. It saves code
hashes and replays each scripted trajectory through the native simulator and the
adapter, checking state equality at every A decision and the same final reward.

## Verified scripted controls

The earlier `environment-contract-check.json` preserves its CPU-only scope.
The later [runtime integration record](environment-runtime-check.json), captured
during `train-bandit-0WF04q1N`, also verifies the actual installed RAGEN registry
class identity. It passes all four native/adapter trajectory checks, deterministic
reset, and invalid-action/timeout checks. This record concerns the existing
installation; it is not an independent clean-install reproduction.

| Fixture | Direct policy | Scripted blocking detour | Interpretation |
|---|---|---|---|
| Slide reproduction | B wins (-1) | A wins (+1) | A useful detour changes the outcome |
| Costly-detour control | A wins (+1) | B wins (-1) | Blocking is not always beneficial |

These are scripted environment controls, not learned-policy results. The audit
also checks deterministic resets and that an illegal parsed action consumes a
turn and reaches the expected timeout in a one-round fixture.

## Existing learned-policy evidence

`docs/final-pilot-results.json` and `docs/final-pilot-results.md` document the
matched local evaluation after `train-blocking-N6WUl30K`. On `blocking_short`, the
base won 1/50 (2%) and the final adapter won 46/50 (92%). Other development fixtures
and the earlier checkpoint comparison are retained in those files. The trajectory
audit and `docs/blocking-case.html` provide action-level case evidence.

These results use fixed development maps and one sequential training chain. They
are not a broad held-out-map benchmark. The exported adapter was evaluated on a
shared cached base with bounded FP32 drift checks, not exact full-checkpoint
replay. Actions-only outputs do not expose an explicit chain of reasoning.

## Extra-credit evidence to submit

- New environment source, adapter, configurations and registry instructions.
- Passing rule/replay tests and the live registry audit JSON.
- A short successful RAGEN training log, reward/loss curves and checkpoint audit.
- Before/after results with their fixed-fixture limitations and a visual case.
- Separate native Bandit experiment for additional environments/baselines; see
  `docs/bandit-extra-credit.md`. Multiple maze layouts alone are not multiple
  independent environments.

The environment extension is already implemented and has prior real RAGEN
training evidence. The new audit makes the contribution easier to reproduce and
review; it does not claim a new training result or guarantee bonus points.
