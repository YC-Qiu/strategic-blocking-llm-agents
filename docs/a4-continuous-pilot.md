# A3 audit and A4 continuous pilot

## Verified A3 evidence

The one-iteration run train-warmup-lrvhQ5LF sampled one win and seven losses. Episode return averaged -0.75. Logged advantage extrema were -0.3535528779 and +2.4748702049, policy-gradient loss 0.2101624589, and actor gradient norm 3.328785345. Actor entropy coefficient and weight decay were zero; reference KL and KL rewards were disabled. Model and optimizer checkpoints were saved. The measured iteration took 139.15 seconds. These results demonstrate a nonzero task-reward training signal, not improved post-update performance.

The captured installed source in a3-source-audit-giwi3xk7 was inspected. Six CPU checks executed the installed function bodies on synthetic tensors; the checks and their limits are recorded in a3-source-audit.json.

- ctx_manager.py stores responses as input_ids[:, 1:], covering almost the full conversation. The generic verl length helper counts the attention mask over that whole field. Thus response_length/mean is not generated-answer length in this adapter. Use train/response_length for the context manager's last-assistant mask length; it still includes the assistant header and fixed answer prefix.
- The limited_multi_turn mask selects the last assistant turn and excludes earlier dialogue and left padding. agent_trainer.py assigns loss_mask to response_mask, and the actor passes that mask to the policy loss. It includes assistant headers and prefix tokens, so this is not exclusively a newly-generated-token mask.
- The trainer sets uid from group_ids and passes episode_ids to GRPO. The GRPO function deduplicates each group/episode pair before computing mean and sample standard deviation. A synthetic batch containing repeated turns and a padding duplicate reproduced the exact logged advantage extrema for seven losses and one win.

These checks explain the suspicious length statistics and support the configured grouping path. They do not reconstruct an unsaved live training batch or prove every rollout correct. The logged PPO KL estimate, about 0.974, warrants a smaller update and continued monitoring. It does not alone identify the cause. Generic per-row score means must not be reported as episode return: repeated turns and padding can change the former.

## Run the five-iteration pilot

~~~bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
bash scripts/run_ragen.sh train-five
~~~

This starts one fresh run from the base model with LoRA and carries its weights through five iterations. It does not resume the preceding warmup checkpoint. Each iteration requests one group of eight warmup_right episodes at temperature 1.0. The learning rate is reduced from 1e-4 to 2e-5; entropy coefficient and weight decay stay zero. Existing micro-batch, SDPA, offload, and four-concurrent-sequence settings are retained. Multi-iteration GPU execution is pending.

Before training and after each update, the same current policy is evaluated greedily on warmup_right and warmup_left, one episode each. Right is the training fixture. Left is an already-inspected mirrored control, not a previously unseen held-out test. This small deterministic evaluation checks behavior changes, not a statistically reliable generalization rate. Validation trajectories are retained in the run's validation directory. Sampled training reward is measured before its corresponding update; greedy validation is after that update (step zero is before training).

Only the final iteration saves a checkpoint, avoiding five large model copies. The new run still needs roughly 2.6 GB for its main model and optimizer, plus transient resources and logs. Earlier checkpoints are preserved. An interrupted run may have no new checkpoint, but its completed console metrics remain available. Do not automatically restart failed training before inspecting the log.

On success, the launcher exports metrics.csv, metrics.json, learning-curves.png, and learning-curves.pdf in the new results/train/train-five-* directory. The four panels show episode reward, actor policy-gradient loss, success fraction, and advantage extrema without smoothing. Missing values are not filled with zero. Nonfinite numeric log metrics are rejected. A plotting error does not mean the model training failed; inspect run.log and the final checkpoint separately.

To regenerate curves or inspect an interrupted run's completed steps:

~~~bash
python scripts/summarize_training.py results/train/train-five-RUN_ID
~~~

Do not expect RL loss to decrease monotonically. Uniform-return groups can have zero advantages, including all-win groups. Inspect reward, actual greedy behavior, clipping, KL, and finite gradients together. Five iterations provide an initial stability and curve check; they do not establish learned blocking. After this run, compare baseline and final validation traces before choosing a longer run or a harder curriculum. No additional environment or reasoning-process bonus is claimed here.
