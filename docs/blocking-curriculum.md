# Mixed blocking curriculum

## Purpose and current status

The new `train-blocking` mode is prepared for the existing Ubuntu RAGEN setup.
Its first GPU run completed as `train-blocking-N6WUl30K`. See
[verified results](final-pilot-results.md) before starting additional experiments.
The mode starts from the step-5 two-step adapter,
adds direct practice on `blocking_short`, and rehearses two-step navigation.
One-step navigation remains a validation-only retention check. All maps remain
development fixtures; training on the blocking fixture removes any basis for
calling its subsequent improvement transfer to an untrained layout.

## Configuration

| Setting | Value |
|---|---|
| Source adapter | `train-two-step-sFYit8xQ/checkpoints/global_step_5/actor/lora_adapter` |
| Base / adapter | Qwen2.5-0.5B-Instruct / LoRA rank 8, alpha 16 |
| Optimizer | Fresh optimizer; adapter warm start, not exact training resume |
| Objective | GRPO, no entropy bonus, no KL reward, no reward shaping |
| Learning rate | 2e-5 |
| Training group 1 | `blocking_short`, 8 sampled episodes |
| Training group 2 | `warmup_two_right`, 8 sampled episodes |
| Training group 3 | `warmup_two_left`, 8 sampled episodes |
| Total budget | 24 episodes per update, 5 requested updates, up to 120 episodes |
| Validation | One greedy episode on each of the five maps before training and after each update |
| Checkpoint | Save at step 5, with the existing early-stop checkpoint repair |
| Inference concurrency | At most 4 sequences; unchanged from previous pilots |

Each GRPO group samples a single initial state so reward normalization does not
mix different maps. The total rollout budget is 50% larger than the two-step
stage. This is not a compute-matched curriculum ablation. Training can still
stop early; a recorded step count alone does not establish normal completion.
The launcher preserves its import/preflight checks, source adapter inspection,
first-collection verification, and checkpoint/curve summarization.

## Start in Ubuntu

After applying the source patch:

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -B -m unittest discover -v

PILOT_ADAPTER="$PWD/results/train/train-two-step-sFYit8xQ/checkpoints/global_step_5/actor/lora_adapter" \
  bash scripts/run_ragen.sh train-blocking
```

Run training only after the tests pass. The launcher automatically runs the
existing RAGEN preflight before training. Use `train-blocking --check-only` if
you want only preflight and configuration composition; preflight itself may
allocate GPU resources. It does not run the five-update training loop.

The command prints a fresh `results/train/train-blocking-*` path. Preserve that
path, the source adapter and every previous result directory. Check for:

- `PASS: first collected actor weights match the supplied warm-start adapter.`
- Nonzero advantages, and separate blocking/right/left training success metrics.
- Validation results on all five maps, including one-step retention.
- A nonempty final checkpoint and adapter, plus `metrics.json` and learning curves.

The plots now include blocking reward/success series. Navigation labels
right/left refer to the two-step maps in this mode. A failed validation map can
still trigger the trainer's existing early-stop rule; do not disable that rule
just to obtain a complete progress bar.

## Evaluate with the corrected seed protocol

The default evaluator now uses `sha256-v2`. It derives each request seed from a
versioned domain, experiment seed, environment ID and turn. Both policies share
the derived seeds. The manifest and per-generation traces record the scheme;
workers receive it explicitly. `legacy-linear-v1` remains available for historical
reproduction. Hashing removes the known arithmetic overlap but does not prove
independence of random streams or eliminate every possible hash collision.

Compare the source checkpoint and the new checkpoint under the same new protocol.
The source can be evaluated after training because its weights are unchanged.
The original two-step source results remain useful historical evidence, but are
not matched sample pairs with SHA256-based results.

```bash
python -u scripts/evaluate_checkpoint.py \
  --training-run results/train/train-two-step-sFYit8xQ \
  --layouts warmup_two_right warmup_two_left blocking_short warmup_right warmup_left \
  --seed 20261111 --seed-scheme sha256-v2 --episodes-per-layout 50

# Replace this value with the path printed by the new training run.
BLOCKING_RUN="results/train/train-blocking-REPLACE_WITH_ACTUAL_RUN"
python -u scripts/evaluate_checkpoint.py \
  --training-run "$BLOCKING_RUN" \
  --layouts warmup_two_right warmup_two_left blocking_short warmup_right warmup_left \
  --seed 20261111 --seed-scheme sha256-v2 --episodes-per-layout 50
```

These two commands run 500 episodes each. If training stopped before step 5,
inspect the saved checkpoints and specify the actual `--step`; do not assume a
step-5 adapter exists. Keep layout order, episode count, sampling parameters and
seed scheme identical across the two checkpoint evaluations.

## Interpretation and next decision

The first question is whether blocking success improves while navigation remains
usable. Inspect both successes and failures, invalid actions, and the initial
block-removal counterfactual. Do not select only attractive trajectories. A
single greedy success is a smoke check; sampled results are the main comparison.
If the blocking reward group repeatedly has identical returns, inspect its raw
rollouts before increasing the training budget.

The current right-map instability is unresolved; continuing training is an
exploratory next experiment, not a demonstrated fix for that instability.
Independent training seeds, maps held out before further tuning, and a matched
training-budget control remain future work. Report explicit limitations rather
than claiming robust strategic reasoning or guaranteed course bonus points.
