# Two-step real rollout check

Tiny FSDP collection verification passed in rollout-sync-audit-_5eg67oq. Both collector paths returned all 28 tensors exactly and reflected the deliberate parameter edit. Captured PEFT settings confirmed that target_modules contained qualified module paths; the corrected guard handles these while retaining complete tensor-key validation. These checks do not verify real vLLM inference.

Run bash scripts/run_ragen.sh train-sync for the next bounded experiment. This uses the real Qwen2.5-0.5B model, fresh rank-8 LoRA, two consecutive training iterations, learning rate 2e-5, and eight sampled warmup_right episodes per iteration. It retains baseline and post-update greedy evaluation on right and left fixtures. Only the final step saves a checkpoint. Existing modes retain their configurations.

Inspect run.log for PILOT_LORA_COLLECT tensors=336 and PILOT_LORA_SYNC tensors=336. The collector checksum should change after a nonzero update, and each vLLM registration must retain exactly the current adapter ID. Check the final checkpoint and compare validation trajectories. Registration and changed source tensors support the transfer path but do not by themselves establish numerical inference equivalence or improved behavior. If advantages remain zero, unchanged checksums alone do not identify a sync defect.

The patched full-parameter collection has passed only tiny-model GPU validation so far. A full-model memory or loading failure must be investigated before extending training. The warnings printed during cleanup of the successful tiny diagnostic do not erase its recorded tensor checks; they also do not prove full-run stability.

This short experiment is not a repeat of five independent launches and does not resume the failed five-step run. Report the new Training log path after completion. Do not extend the run until the real synchronization evidence and final checkpoint have been checked.
