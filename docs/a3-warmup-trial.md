# A3 sampled probe result and warmup training trial

The base-model probe a3-probe-9cdws2a6 completed 28 episodes: one win and 27 losses. All episodes resolved. Only warmup_right had mixed terminal returns: [-1, -1, -1, +1], with population variance 0.75. Each of the other six layouts had four losses. These are exploratory samples, not reliable population win-rate estimates.

The winning response was <answer>move right</answer>, reaching the adjacent goal in one action. The three failures on this map were: two invalid responses converted to waits; moving down twice; and moving down followed by blocking down. This demonstrates a reachable positive reward under sampling, not a learned policy or successful strategic blocking.

Across 115 recorded decisions, 15 parsed actions were invalid. The answer-tag diagnostic found no errors, illustrating that correct tags do not guarantee a valid action. The short blocking map had two episodes with an obstacle placement, but neither won. See a3-probe-2026-11-08.json for curated evidence and source hashes. Raw artifacts remain under results/ and are excluded from Git.

## Next experiment

Run the existing launcher in its new train-warmup mode:

~~~bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
bash scripts/run_ragen.sh train-warmup
~~~

This starts fresh base-model LoRA training for one iteration, using one group of eight sampled warmup_right episodes. It retains temperature 1.0, the action-only prompt, terminal rewards, learning rate, and the existing memory/offload setup. Eight episodes are requested, with at most four concurrent inference sequences. A larger group improves the opportunity to observe mixed returns, but does not guarantee it; the evaluation samples are not replayed as training data.

For this diagnostic only, entropy coefficient and optimizer weight decay are both zero. Reference KL and KL reward are already disabled by the existing launcher. This reduces alternative causes of updates when advantages are zero. These settings are an experimental change and must be disclosed when comparing this trial to the earlier smoke tests. The previous train-one mode retains its original configuration.

The launcher runs the existing preflight checks and saves config.yaml, composed.yaml, dependencies.txt, preflight.log, run.log, and the main checkpoint beneath a new results/train/train-warmup-* directory. It does not resume the earlier one-iteration checkpoints. Its automatic standalone adapter export remains subject to the previously documented issue. The new mode has passed configuration-generation checks locally, but GPU training has not yet been verified.

## How to judge the run

- First check the actual sampled success count and episode returns. A uniform-return batch can still produce zero advantages; do not call this a trainer failure solely from zero values.
- If there are both wins and losses, inspect critic/advantages/min and max for a nonzero range. Check finite actor gradients and loss, as well as the saved update.
- Inspect actual trainer group assignment before treating the signal as a validated GRPO group. The earlier evaluation probe only grouped identical observations after sampling.
- The mean actor/pg_loss can be near zero through cancellation even when gradients are useful. Neither a nonzero scalar loss nor changed weights alone proves learning.
- If mixed returns still yield zero advantages, investigate trajectory-to-turn mapping and GRPO grouping before increasing training length.

A3 remains in progress until the training signal is verified. A4 requires continuous training and reliable metrics, and later evaluation must test matched settings and unseen maps. Navigation improvement on this one map would not establish blocking strategy or generalization. No thinking-process claim follows from this action-only protocol.
