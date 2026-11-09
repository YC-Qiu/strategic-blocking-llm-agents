# First paired evaluation and the next curriculum pilot

## Evidence and current status

Training run: `results/train/train-five-PJTvqpWO`, checkpoint step 5.
Evaluation: `results/paired-eval-588ollqx`, sampling seed `20261108`.
Both policies used the same cached Qwen2.5-0.5B-Instruct base, with and without
the exported LoRA. Temperature was 1.0, top-p 1.0 and top-k -1. Initial prompts
and request seeds matched. Each policy completed 50 episodes per fixed map.

| Map | Base wins | Adapter wins | Change in success rate |
|---|---:|---:|---:|
| warmup_right | 17/50 | 25/50 | +16 percentage points |
| warmup_left | 1/50 | 3/50 | +4 percentage points |
| blocking_short | 1/50 | 0/50 | -2 percentage points |

The training-fixture result is encouraging but not a robust improvement claim.
For the right fixture, 12 paired samples changed from a base loss to an adapter
win and 4 changed in the opposite direction. An exploratory two-sided exact
McNemar test gives p=0.0768, without correction for inspecting multiple maps.
This describes one training run and one evaluation seed, not training-seed
variability. These development fixtures are not a held-out generalization set.

The base-weight audit found bounded differences in all 168 LoRA target matrices,
with maximum absolute difference 5.96046448e-08. The audit passed its declared
1e-7 absolute bound; other frozen weights required exact equality. This evaluation
measures the exported adapter's effect on a shared cached base, not exact replay
of the full FP32 checkpoint.

The framework, continuous pilot, loss/reward curves, checkpoint export and first
matched evaluation now have observed evidence. Strategic learning and a robust
baseline improvement remain unproven. No new teammate/Colab completion is implied.

## Behavior findings

| Metric | Base | Adapter |
|---|---:|---:|
| Invalid parsed actions / actions with known validity, all maps | 42/364 (11.5%) | 82/334 (24.6%) |
| First action is move right on warmup_right | 8/50 | 15/50 |
| Episodes with an executed block on blocking_short | 24/50 | 6/50 |
| First action is block up on blocking_short | 2/50 | 0/50 |

Invalid-action rates are descriptive action-level rates. Turns within an episode
are correlated, and episode lengths differ between policies. They are not
independent per-episode success measurements.

The adapter more often makes the immediately winning right move on its training
fixture, while action validity deteriorates and blocking becomes less frequent
in this sample. This is consistent with a narrow behavior change; it does not
establish the cause or prove that the model learned a direction-independent rule.

## Replayed case study

All 300 exported trajectories were replayed through the local environment. Each
pre-action observation, executed action and final outcome matched the records.

The base model's `blocking_short` episode 136 begins with `block up`, forcing the
BFS opponent onto a detour. The model subsequently wastes several moves but
still wins after 11 actions. Replaying the same action list with only the first
action changed to `wait` makes B win after the third A action. This counterfactual
supports the contribution of the block in that specific trajectory.

This is a base-model success, not an RL success. The action-only protocol contains
no explicit reasoning trace, so the case cannot establish the model's intent or
internal thinking process. The adapter's zero observed blocking wins remain a
negative finding worth including in the report.

## Next experiment: balanced navigation warmup

Run a fresh five-update pilot from the original base using two separate GRPO
groups: eight samples from `warmup_right` and eight from `warmup_left` per update.
Keep the existing model, prompts, terminal rewards, learning rate 2e-5, LoRA
settings, greedy validation and checkpoint behavior. Preserve the original run.

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
bash scripts/run_ragen.sh train-balanced
```

This produces 16 training episodes per update, versus 8 previously, while keeping
four concurrent inference sequences. The episode budget is doubled to retain
eight samples per GRPO group. This is a curriculum screening experiment, not a
budget-matched causal comparison. Runtime may increase substantially. It does not
resume the direction-specific adapter and does not yet train blocking behavior.

Inspect both map-specific training success series. If the left group continues
to contain only losses, nonzero aggregate advantages may come entirely from the
right group; they do not show that leftward navigation received a useful reward
signal. Do not automatically extend training in that case. Revisit exploration
or an explicitly labeled demonstration-assisted curriculum instead.

After the run, evaluate its saved step-5 adapter using the same paired evaluation
protocol. Replace the example path with the actual new run directory:

```bash
python -u scripts/evaluate_checkpoint.py \
  --training-run results/train/train-balanced-REPLACE_WITH_RUN_ID \
  --episodes-per-layout 50
```

Proceed toward blocking training only after checking navigation across both
directions, action validity, and whether each training group has reward variation.
If results look better, confirm with more evaluation samples and independent
training seeds. If validity still deteriorates, address the action-format failure
as a separate experiment. A fixed reward can credit earlier invalid actions in a
later winning episode; this is a hypothesis to investigate, not an established
explanation of the observed regression.
