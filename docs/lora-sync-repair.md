# Pilot LoRA collection and rollout refresh repair

The installed tiny-FSDP diagnostic rollout-sync-audit-b3i01da0 expected 28 adapter tensors. Both layered and full collection returned zero tensors, and neither reflected a deliberate parameter edit. This confirms a collector defect for the tested wrapping layout. It supports investigation of stale rollout behavior but is not a direct measurement of the earlier full model's vLLM state.

The captured synchronous vLLM implementation also registered successive adapters under new IDs while generation selected the first listed ID. That path did not guarantee use of the newest adapter. The repair retains exactly one current adapter and rejects empty or duplicate input tensors before replacing it. Registration is checked through list_loras; real-engine inference equivalence is still unverified.

For this single-GPU Qwen2 pilot only, both adapter-collection entry points now expose all FSDP parameters recursively, select the named default LoRA A/B parameters directly, and clone only those tensors to CPU. It validates the complete expected layer/projection key set, shapes, rank, finite values, and supported plain-LoRA configuration. The tiny model must yield 28 tensors and the 24-layer pilot model 336. Unsupported configurations fail explicitly. The collector logs the count and a checksum so later rollouts can be checked for changed weights. It also feeds the existing automatic export path; actual full-model export validation remains pending.

This full-parameter exposure may have a higher transient GPU memory cost than a working layered implementation. Only adapter tensors are cloned to CPU; this is not a general distributed collector replacement. Installed source hashes are checked before editing, and original files are backed up. Reapplying the same patch is safe; changed upstream code requires review.

The same repository patch includes the previously prepared early-stop checkpoint preservation and missing-metric reporting fixes. These do not recover the lost checkpoint from train-five-jDIUUSmx and do not disable early stopping.

Local validation: six CPU tensor-extraction guard tests pass; captured patched code parses; a fake-engine test executes the captured synchronous update method and checks successive replacement, singleton adapter selection, and rejection of empty/duplicate input without destroying the prior adapter. These tests do not substitute for the next GPU collection diagnostic or actual vLLM inference.

After applying the repository patch, run python scripts/patch_ragen.py --ragen-dir "$HOME/RAGEN", then python scripts/diagnose_rollout_sync.py. Do not restart five-step training until both GPU collection cases pass. Next verify a short real actor-to-vLLM update using the checksum/count logs and matched outputs before extending the run.
