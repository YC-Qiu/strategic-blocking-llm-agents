# Two-step transfer evaluation and navigation continuation

The balanced adapter transfers only partially from one-step to two-step navigation.
It also produced one valid short-blocking success, which is a case study rather
than evidence of a reliable learned blocking policy.

Source: `results/paired-eval-ikmwigjl`, balanced checkpoint
`train-balanced-jVnNYJRu/global_step_5`. Each policy completed 50 episodes per map
with sampling seed 20261109. All 300 trajectories were replayed with matching
pre-action states, executed actions and outcomes. These are previously inspected
development fixtures, not a held-out map benchmark.

| Fixture | Base wins | Balanced adapter wins |
| --- | ---: | ---: |
| warmup_two_right | 5/50 | 9/50 |
| warmup_two_left | 1/50 | 9/50 |
| blocking_short | 0/50 | 1/50 |

## Behavior diagnosis

On the right fixture, the adapter made the correct first move in 28/50 episodes,
versus 8/50 for the base. Nevertheless, it won only nine episodes. Frequent failed
executed sequences were `right, left, right` (10 cases) and `wait, right, left`
(10 cases). On the left fixture, the adapter started with `move left` in 22/50
episodes versus 3/50 for the base, but often reversed direction afterward.
This supports a distinction between selecting the initial direction and reliably
following the route over multiple observations. It does not prove a specific
internal representation or reasoning failure.

Invalid-action rates were 24/145 (16.6%) and 26/144 (18.1%) for the adapter on the
two navigation maps. Legal but unproductive reversals also contributed to losses,
so output-format correction alone would not address all observed failures.

Adapter episode 101 on `blocking_short` executed:

```text
block up -> move down -> move right -> move right -> move right
```

It won in five actions. A replay changing only the first action to `wait` lost
after the third A action. This demonstrates that the initial obstacle mattered
for that trajectory. The preceding evaluation seed yielded zero adapter blocking
wins, and the base has also produced a blocking win in earlier sampling. One
success cannot establish an RL-induced improvement. No reasoning text was elicited.

## Next training run

Use the saved balanced adapter as the starting policy, then train only the two-step
right/left fixtures for five updates, eight samples per group. The base remains
Qwen2.5-0.5B-Instruct; terminal rewards, learning rate 2e-5, rank-8 LoRA and other
optimizer settings stay unchanged. This uses 80 additional sampled episodes.

This is an adapter warm start with a fresh optimizer, not exact checkpoint resume.
Do not join its curves to the previous run as if optimizer state was continuous.
The launcher validates the supplied adapter, records its tensor digest and checks
the first collected actor weights against that digest after training. A mismatch
invalidates the claimed warm start. The new loading path still requires its first
real GPU execution; CPU checks do not establish GPU compatibility.

Greedy validation includes both two-step maps plus the original one-step maps as
retention checks. One-step fixtures are not part of this run's training batch.
The existing low-success early-stop rule still applies and can trigger after the
fifth update; final checkpoint saving remains enabled.

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
PILOT_ADAPTER="$PWD/results/train/train-balanced-jVnNYJRu/checkpoints/global_step_5/actor/lora_adapter" \
  bash scripts/run_ragen.sh train-two-step
```

The log should report loading the pretrained actor adapter and finish with:

```text
PASS: first collected actor weights match the supplied warm-start adapter.
```

Review the per-map rewards and saved checkpoint before spending more training
budget. The proposed follow-up evaluation is:

```bash
python -u scripts/evaluate_checkpoint.py \
  --training-run results/train/train-two-step-REPLACE_WITH_RUN_ID \
  --layouts warmup_two_right warmup_two_left blocking_short warmup_right warmup_left \
  --seed 20261109 --episodes-per-layout 50
```

Keeping the first three layouts in the same order preserves their environment-ID
seed assignment from this transfer evaluation. The added one-step maps measure
retention against a matched base control. Interpret them as a new sampling block,
not a paired repetition of the earlier one-step evaluation. Blocking training
and independent training-seed replication remain later experiments.
