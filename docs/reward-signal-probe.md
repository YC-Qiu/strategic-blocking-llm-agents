# A3: Diagnose the terminal reward signal

Status: local fixture and evidence-accounting tests pass. The sampled GPU probe completed with one informative diagnostic group out of seven. See [the results and pending training trial](a3-warmup-trial.md). A3 remains in progress; improved performance has not been demonstrated.

## Why this comes before another training run

Both completed training runs sampled four losses, with zero GRPO advantages and zero policy-gradient loss. Repeating the same configuration has not demonstrated reward-driven learning.

The two saved greedy before-RL episodes each contain eight legal, correctly tagged actions. Both start with two waits, then move right twice and alternate down/up without blocking. Therefore formatting alone does not explain those failures. Sampled training also showed illegal actions; greedy and sampled behavior must be analyzed separately. See a3-baseline-diagnosis.json for the extracted greedy evidence.

The next experiment samples the base model without updating its weights. We check whether repeated trajectories from the same initial observation can receive different terminal rewards. Successful scripted policies establish that a fixture is solvable; they do not establish that the model will discover those policies.

## Candidate fixtures

| Layout | Scripted check | Purpose |
| --- | --- | --- |
| warmup_right / warmup_left | Direct movement wins in one A action; waiting loses | Basic goal-directed movement in both directions |
| warmup_two_right / warmup_two_left | Direct movement wins in two A actions; waiting loses | Slightly longer navigation |
| blocking_short | Direct movement loses; block up followed by direct movement wins in five A actions | Short blocking decision |
| slides | Original reproduction | Existing difficult reference |
| costly | Original costly-detour control | Existing reference where direct movement wins |

The five new maps are authored curriculum candidates in curriculum.py. Rules, B's policy, and terminal rewards remain unchanged: win +1, loss -1, timeout 0. These fixtures are not held-out evaluation data. Reset seeds do not create new maps. A win on blocking_short alone does not demonstrate strategic generalization or learned timing.

## Run on the existing Ubuntu installation

Use the same patched RAGEN checkout and conda environment that completed A1. Install this project's source patch first, then run:

~~~bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -B -m unittest -q
python scripts/probe_reward_signal.py
~~~

The default requests seven layouts, four episodes each, temperature 1.0, and seed 42: 28 episodes total. It uses Qwen/Qwen2.5-0.5B-Instruct, the existing action-only prompt, two-turn context, and the evaluation entry point. It does not load the recovered LoRA adapter or call the trainer. No optimizer or weight update is involved. This preserves a base-model diagnostic before selecting a training curriculum.

The rollout uses a 2,048-token model limit, 128 response tokens, eager execution, at most four concurrent sequences, and GPU memory utilization 0.5. This configuration completed the 28-episode probe on the existing local GPU installation. These settings are not a guarantee against out-of-memory errors. Close another model process before starting.

Configuration-only check, without starting RAGEN:

~~~bash
python scripts/probe_reward_signal.py --prepare-only
~~~

To point to another existing checkout, add --ragen-dir /path/to/RAGEN. To reduce the initial experiment, add --layouts warmup_right warmup_left blocking_short. Keep group size at least two. Record any change to sampling settings; do not silently combine runs as one experiment.

The program prints its output directory and log path before sampling. It writes RAGEN output to run.log; inspect that file from another terminal while the process runs. A RAGEN failure stops the program and identifies the log. Configuration preparation and unit tests do not substitute for completing this GPU run.

## Evidence and interpretation

Each unique results/a3-probe-* directory contains:

- config.yaml and manifest.json: requested settings and project source hashes.
- dependencies.txt, ragen-revision.txt, ragen-working-tree.patch, and ragen-status.txt: installed package versions and checkout provenance. Untracked file contents are not included in the patch.
- run.log and rollout.pkl: raw output from this invocation.
- episodes.jsonl: observations, responses, actions, metrics, and outcomes retained from the raw rollout.
- summary.json: per-layout return lists, outcome counts, reward variance, format diagnostics, parsed-action validity, and blocking counts.

The analyzer verifies the initial observation and expected episode count. A completed episode's return must agree with its terminal outcome. A trajectory cut off by the manager is marked unresolved, not silently converted into a loss or timeout. Raw JSON is saved before strict summary checks, so a mismatch can be investigated.

An informative diagnostic group has only resolved episodes and nonzero terminal-return variance. The grouping is reconstructed from identical initial observations. It does not verify the trainer's internal GRPO group IDs or advantage calculation; summary.json records this limitation explicitly. The answer-tag check is a diagnostic heuristic, not a reimplementation of RAGEN's parser. Missing adapter validity information is reported as unknown rather than counted as valid.

| Observed result | Next action |
| --- | --- |
| Mixed terminal returns on a small layout | Inspect raw successful and failed trajectories; use that layout as a candidate for a short training trial |
| All wins on a layout | Check a harder layout; identical rewards still provide no within-group GRPO signal |
| All losses | Inspect legal actions, responses, movement, and context before changing sampling or curriculum |
| Unresolved episodes or format failures | Diagnose the manager/format path before interpreting reward variance |
| Only navigation layouts show mixed returns | Start navigation warmup; retain blocking as an unverified next stage |

Before claiming A3 complete, run a training trial on the selected curriculum and verify actual trainer grouping, nonzero advantages, finite policy-gradient loss, and stable updates. A4 still requires continuous training and reliable metrics; A5 still requires training and model selection. Matched before/after evaluation also remains pending. This probe supplies neither trained improvement nor a loss curve. The current action-only protocol also supplies no explicit reasoning text for a thinking-process case study.

## Validation performed here

The full local suite passes 28 tests: 10 existing environment/adapter tests, seven adapter-export guards, and 11 new curriculum/probe tests. The new checks cover scripted win/loss mechanics, native/adapter replay, fixed-seed behavior, mixed versus uniform returns, unresolved trajectories, episode count and state mismatches, reward/outcome consistency, and the generated sampling configuration. GPU probe execution subsequently passed; the warmup training trial remains pending.
