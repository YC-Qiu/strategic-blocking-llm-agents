# Frozen prospective transfer test

The declared test is `prospective-transfer-v1`, with eight maps, 50 episodes per
map and policy, and experiment seed 20261112. Map definitions and adapter hashes
are in `evaluation/frozen-transfer-v1.json`. The loader checks the exact file
SHA256; editing a map requires a new suite version. The existing training modes
do not select these maps.

## Questions fixed before policy evaluation

| Family | Maps | Question |
|---|---:|---|
| Symmetry | 2 | Does blocking behavior survive a horizontal mirror or vertical flip? |
| Longer blocking corridors | 2 | Does it transfer to a longer race and its mirror? |
| Direct-win controls | 2 | Can it finish directly when a blocking detour is unnecessary? |
| Three-step navigation | 2 | Does navigation extend beyond the two-step training maps? |

Rule search verifies that every map has a winning action sequence. A direct
shortest-path policy loses on the four blocking maps and wins on the other four.
The rule-check paths are for validation only and are not included in model
prompts. These are authored variations of familiar geometry; success would not
establish broad out-of-distribution strategic reasoning.

## Preferred split of work

YC freezes and packages the source and adapters, and prepares the local pilot
report. Prakeerth runs the [Colab notebook](../notebooks/colab-transfer.ipynb)
following [the handoff guide](colab-transfer.md), returning raw episodes and
complete provenance. No additional RL training is needed for this experiment.

## Optional local RAGEN path

Use this only if the team chooses to evaluate the same maps in the existing
RAGEN environment. It is a separate backend result from the Colab experiment.
The two commands below each evaluate base plus one adapter, for 1,600 episodes
total. Base is repeated; do not count the duplicate run as a new training seed.

```bash
LAYOUTS=(frozen_block_mirror frozen_block_vertical frozen_block_wide \
  frozen_block_wide_mirror frozen_direct_near frozen_direct_near_mirror \
  frozen_navigation_three frozen_navigation_three_mirror)
for RUN in train-two-step-sFYit8xQ train-blocking-N6WUl30K; do
  python -u scripts/evaluate_checkpoint.py \
    --training-run "results/train/$RUN" --layouts "${LAYOUTS[@]}" \
    --seed 20261112 --seed-scheme sha256-v2 --episodes-per-layout 50 || break
done
```

Report all eight map results, including negative results, and retain the exact
protocol. Do not tune prompts, maps or checkpoints after looking at these scores.
Further development requires a fresh evaluation set. Independent training
replications and compute-matched ablations remain separate future work.
