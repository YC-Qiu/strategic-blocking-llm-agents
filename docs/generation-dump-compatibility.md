# Generation dump compatibility

The first train-five attempt stopped during baseline validation, before any training iteration, because RAGEN omitted the gts argument now required by the installed verl generation writer. The five-step launcher enabled validation trajectory saving, exposing this interface mismatch that earlier one-step runs did not exercise.

The compatibility patch adds gts=[None] * len(inputs) at both known RAGEN dump call sites, using sample_inputs for validation. Strategic Blocking has no reference answer string, so JSON null is intentional. Inputs, outputs, scores, and evaluation remain unchanged. This does not change the model objective or reward function.

The existing launcher automatically invokes the updated scripts/patch_ragen.py. The patch verifies the installed signature and both call sites, preserves already patched sources, rejects unexpected existing ground truths, and saves agent_trainer.py.before_generation_gts.bak before editing the separate RAGEN checkout.

Validation: four regression tests passed. Both patched call expressions from the captured installed RAGEN source were executed against the captured installed verl writer; two JSONL records retained their outputs and scores and had null ground truths. The complete GPU validation/training loop has not been rerun here.

After applying this repository patch, rerun bash scripts/run_ragen.sh train-five. A new output directory is expected; the failed attempt and its log are retained. This repair covers the missing gts interface argument. The optional training-rollout dump path is not enabled by this pilot and has not been tested end to end.
