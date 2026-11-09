# Compare the base policy with the five-update checkpoint

The next experiment is a matched sampled evaluation, not another training run.
The five-update pilot completed its actor updates and saved a checkpoint, but
its greedy validation success stayed at zero. Nonzero advantages and changing
LoRA weights do not establish better task performance.

## Run in Ubuntu

Use the existing `ragen` Conda environment and the patched RAGEN checkout.
No new packages, model downloads, or optimizer updates are requested.

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -u scripts/evaluate_checkpoint.py \
  --training-run results/train/train-five-PJTvqpWO \
  --episodes-per-layout 50
```

This evaluates two policies on three fixed maps, for 300 episodes total:

| Map | Purpose |
|---|---|
| `warmup_right` | Check improvement on the training fixture. |
| `warmup_left` | Check whether behavior transfers to the mirrored fixture. |
| `blocking_short` | Check for successful strategic blocking behavior. |

The script verifies the automatic LoRA export exactly against the checkpoint.
It audits every frozen checkpoint tensor against the cached base. Outside LoRA
target matrices, equality must be exact. For FP32 LoRA target matrices only, it
permits at most `1e-7` absolute difference, without a relative tolerance. Repeated
LoRA merge/unmerge operations can introduce small FP32 roundoff. Shapes and finite
values are checked, and larger or unexpected changes stop evaluation. Raw and
rollout-precision differences are saved even when the audit passes.

Both policies use the same cached original base, with and without the exported
adapter. This measures the deployed adapter's effect; it is not exact replay of
the full FP32 checkpoint. Even very small raw differences can cross BF16 rounding
boundaries near zero, so the report does not assert BF16 equality unless verified.
The script writes a normalized adapter copy under the new results directory and
leaves the source checkpoint intact. A missing cache stops evaluation.

The model processes run sequentially to avoid loading two models into GPU memory.
The adapter process runs first so an adapter-loading error fails promptly.
Both policies use the same cached base snapshot and enable the same LoRA-capable
vLLM engine. Only the trained policy receives a LoRA request. Each generation
checks the registered adapter IDs.

The configuration retains the recorded training instructions, context window,
turn limit and action parser. Evaluation uses temperature 1.0, top-p 1.0 and
top-k -1. Distinct request seeds depend on environment ID and turn number, so
finishing one episode does not shift the random seeds assigned to other episodes.
Initial prompt hashes and seeds must match across policies. This does not promise
bitwise reproducibility across GPU types or library versions.

## Outputs and interpretation

The script prints a fresh `results/paired-eval-*` directory. Read:

- `comparison.md` and `comparison.csv`: success rates, 95% Wilson intervals,
  observed returns, unresolved episodes and action validity.
- `comparison.json`: the same metrics plus executed-action distributions.
- `case-index.json`: the first available example of each outcome per model/map.
- `base/episodes.jsonl` and `adapter/episodes.jsonl`: raw trajectories.
- `base-weight-audit.json`: per-tensor raw differences, differences after conversion
  to the rollout dtype, and acceptance limits. Review this before reporting results.
- `model-verification.json`, `manifest.json`, `dependencies.txt` and generation
  traces: weight verification, source identities and sampling provenance.
- `adapter.log` and `base.log`: inference logs for diagnosing a failed run.

If a process fails, inspect its log. Do not interpret a partial run as a completed
comparison. Existing results are never overwritten. An unresolved episode is
reported separately and is not relabeled as a loss.

Use results to choose the next experiment:

1. If the trained policy appears better only on `warmup_right`, treat this as a
   candidate training-fixture improvement. Repeat training seeds and evaluate a
   broader curriculum before claiming a robust gain.
2. If differences are small or uncertain, inspect action distributions and failure
   cases before increasing the training budget.
3. If blocking succeeds, replay those trajectories and check the blocking action's
   contribution. Success alone does not establish a deliberate blocking strategy.

These fixtures have been inspected during development. They are not a held-out
generalization benchmark. Fifty samples per policy/map can still yield wide
intervals. The intervals quantify rollout sampling uncertainty for fixed policies,
not variability across training seeds. The action-only protocol does not produce
explicit reasoning traces.

## Checks available without a GPU

```bash
python -B -m unittest test_checkpoint_evaluation -v
python scripts/evaluate_checkpoint.py \
  --training-run results/train/train-five-PJTvqpWO --prepare-only
```

The CPU tests check seed alignment after episodes finish, adapter-registration
failure handling, configuration preservation, uncertainty intervals and summary
consistency. Prepare-only checks configuration construction; it does not verify
weights or run inference. The GPU path must still be exercised in Ubuntu.
