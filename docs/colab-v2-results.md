# Colab v2 transfer evaluation

The downloaded run contains 1,200 unique episodes: eight frozen layouts, three policies, and 50 episodes per layout-policy pair. All recorded trajectories replay exactly against the delivered environment and protocol. The summary matches the raw records.

`before` is the two-step curriculum adapter (`train-two-step-sFYit8xQ`); `after` adds the blocking curriculum (`train-blocking-N6WUl30K`). Both are evaluated on the same cached base revision.

| Layout | Base | Before blocking training | After blocking training |
|---|---:|---:|---:|
| frozen_block_mirror | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) |
| frozen_block_vertical | 0/50 (0%) | 0/50 (0%) | 3/50 (6%) |
| frozen_block_wide | 1/50 (2%) | 10/50 (20%) | 40/50 (80%) |
| frozen_block_wide_mirror | 0/50 (0%) | 0/50 (0%) | 0/50 (0%) |
| frozen_direct_near | 26/50 (52%) | 40/50 (80%) | 48/50 (96%) |
| frozen_direct_near_mirror | 17/50 (34%) | 21/50 (42%) | 14/50 (28%) |
| frozen_navigation_three | 1/50 (2%) | 2/50 (4%) | 2/50 (4%) |
| frozen_navigation_three_mirror | 0/50 (0%) | 5/50 (10%) | 1/50 (2%) |

## Validity checks

- Source, protocol, suite, and adapter hashes match the delivered v2 bundle.
- All 1,200 episodes and all request seeds, prompts, parser decisions, rewards, actions, and final states were replayed or reconstructed. Initial prompts match across policies for all 400 layout/episode pairs.
- The full-run cumulative format gate passes at every episode boundary. This gate is per policy, not per map.
- The manifest embeds a passing smoke report (zero errors across 134 requests). Original smoke files were not included in this download and were not independently replayed.
- Recorded completions were reused; this CPU audit does not independently verify GPU generation or tokenizer decoding.

| Policy | Wins across eight maps | Format errors / requests | Parsed illegal actions | Unresolved episodes |
|---|---:|---:|---:|---:|
| base | 45/400 | 34/1492 (2.28%) | 344 | 1 |
| before | 78/400 | 29/1565 (1.85%) | 175 | 0 |
| after | 108/400 | 32/1560 (2.05%) | 55 | 1 |

## Interpretation

The main positive result is limited transfer to `frozen_block_wide`: base 2%, before 20%, and after 80%. Direct near-goal performance also improves to 96%. This is not broad transfer: the mirrored wide map remains at 0%, the vertical map reaches only 6%, and navigation maps remain weak. The mirrored direct control falls from 42% before to 28% after.

For the 40 after-policy wins on the wide map, replacing the first executed block with wait gives {'B': 40}. The remainder of each recorded action sequence is held fixed. This supports the importance of the block within these trajectories, not a claim about how a re-queried policy would recover.

The wide mirrored map still has local format-error rates of 11.6% (base), 11.1% (before), and 11.5% (after), despite passing the cumulative policy-level gate. Keep these failures and the two unresolved episodes in the reported denominator; do not reinterpret them as resolved losses or silently remove them.

## Scope and limitations

This is Transformers/PEFT FP16 SDPA evaluation on a T4, not exact RAGEN/vLLM checkpoint replay. The maps were frozen before v1, but v2 repairs followed inspection of v1 failures on those maps; this is not a newly blind held-out evaluation. There is one training chain and one evaluation seed. Per-map Wilson intervals are available in the accompanying JSON; they describe sampling uncertainty for each fixed policy, not variation across training seeds. No explicit reasoning traces were elicited, so trajectory behavior should not be described as a recorded thinking process.

## Next steps

1. Preserve the v1 failure and this v2 result as separate protocol records. Include both the wide-map success and mirror-map failure in the report.
2. Consolidate the separate native RAGEN Bandit evidence for the additional-environment/baseline bonus. The local run `train-bandit-0WF04q1N` already contains COMPLETE, learning curves, and a report with five updates and matched evaluation. Its report shows no improvement in high-arm choice (50.0% base versus 46.5% adapter). The Bandit raw trajectories were not independently audited in this Colab audit.
3. If further curriculum work is pursued, treat these eight maps as development data after inspection and freeze a new evaluation suite before training. Do not retroactively change v2 parsing or its acceptance rule.
