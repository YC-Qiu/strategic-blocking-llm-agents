# Five-step result and follow-up

Run train-five-jDIUUSmx recorded five actor updates, each with sampled success 1/8 and mean episode reward -0.75. Both greedy validation fixtures failed at baseline and every update. Baseline and final validation output text was identical. This run provides measured curves but does not demonstrate improved behavior.

The trainer's hard-coded low-success rule triggered on validation five and exited before checkpoint saving and final metric collection. No final checkpoint exists. Advantage metrics are present and nonzero for steps 1-4; step 5 is missing these metrics, not known to have zero advantages. The earlier reporting script obscured this distinction.

The compatibility repair preserves a completed update's checkpoint and advantage metrics before this particular early-stop exit. It does not disable early stopping. The reporting repair distinguishes missing metrics from zeros and warns about early stopping and absent checkpoints. This does not recover the missing model from the previous run.

Before another full training run, inspect LoRA parameter collection and the actor-to-rollout synchronization path. The prior empty automatic adapter export and unchanged sampled behavior are reasons to investigate, not proof that synchronization failed. A tiny FSDP collection diagnostic compares layered and full collection against known tensors and tests whether a deliberate parameter edit is visible. This isolates collection only; vLLM-side synchronization still requires its own verification.
