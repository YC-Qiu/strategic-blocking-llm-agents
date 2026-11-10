# Colab transfer v2: format-interface repair

## What failed in v1

The preserved `colab-transfer-mkvz25_e` run completed 1,200 episodes, all won by B.
Raw episodes contain 4,197 format errors in 4,200 requests. Base executed 1,399
waits and one block-down action; before and after each executed 1,400 waits.
These scores do not measure the intended policy comparison. COMPLETE in v1
proved loop completion, not format validity. See `colab-v1-format-audit.json`.

The captured installed RAGEN source and the two recorded training configurations
confirm these interface mismatches:

- RAGEN appends `<answer>` to the assistant prompt and restores that prefix
  before parsing the generated completion. V1 did neither.
- RAGEN prefixes the system instruction with `You're a helpful assistant. `.
- RAGEN includes turn numbers, remaining action budget, a format reminder,
  a 100-word/token response reminder, and previous rewards.
- Its context window of two includes the current state: one previous turn,
  rather than v1's two previous turns plus the current observation.
- An empty parsed action list consumes a manager request but does not advance
  the environment. V1 passed a sentinel to the environment, which executed wait.

These are confirmed implementation differences. A successful v2 GPU smoke run
is still required to show that the repair resolves the observed failure. The v1
record did not retain raw token IDs, so decoding cannot be fully reconstructed.

## Fixed protocol decisions

V1 code, protocol, map suite and original outputs remain unchanged. V2 uses the
same model revision, adapter hashes, eight maps, seed inputs and 50 episodes per
map per policy. Its seed domain remains `hf-colab-v1` to preserve request seeds
at matching indices; trajectories may diverge after any action or retry.

The primary parser ports the captured RAGEN behavior: first answer-tag match,
removal of its listed control strings, separator splitting, and truncation to
one action. A generated closing tag is required; bare legal actions are not
accepted. Restoring the opening tag already present in the prompt is not a
bare-action fallback. Any future permissive-parser experiment needs a separate
protocol version, and cannot rescore v1's changed trajectories as policy wins.

Malformed responses leave both runners unchanged, earn zero reward and consume
one of 32 manager requests. A tagged but illegal action executes wait and
advances the environment. Exhausted manager budgets are unresolved, not B wins.
The prompt budget aborts on overflow instead of silently truncating. FP16 SDPA
on Colab remains a separate backend from local BF16 vLLM; no exact sampling or
full-checkpoint replay claim is made.

## Run instructions

1. Open `notebooks/colab-transfer-v2.ipynb` in a fresh GPU runtime.
2. Upload `colab-transfer-v2.zip` into `MyDrive/strategic-blocking-transfer/`.
3. Run setup. Optional torchao is removed because this non-quantized evaluation
   does not use it; its prior version is recorded. Colab's CUDA Torch is retained.
4. Run verification and the 45-episode smoke test on old development maps.
5. Each policy separately must have at least 15 requests and at most 5% format
   errors. No win-rate threshold is imposed. Failure blocks the full run.
6. Run the final cell only after smoke passes. It runs 1,200 episodes and checks
   the smoke evidence against identical source, tokenizer, adapters and runtime.
7. Preserve and share the entire output directory, including setup, dependencies,
   manifest, episodes, summary, comparison and COMPLETE (or FAILED.json).

The full run aborts if a policy exceeds 5% format errors after at least 30 requests,
checked at episode boundaries. Partial results remain on Drive. Do not automatically
rerun or tune the prompt after seeing transfer scores. The selected threshold is
an operational quality gate, not a statistical guarantee of protocol validity.

## Evidence and validation

`evaluation/colab-ragen-contract-v2.json` records hashes of the captured RAGEN
sources, relevant training configuration values, and the five ported text methods.
CPU tests cover prefill/restoration, history boundaries, retry state preservation,
illegal actions, bounded failures, and smoke evidence integrity. The runner also
checks tokenizer delimiter round-trips and retains input/output token IDs.

The repaired protocol was prepared after inspecting v1 outputs on these maps.
Report this as a disclosed protocol repair, not a newly blind held-out experiment.
V2 GPU behavior and policy scores remain unverified until the notebook is run.

To rebuild the private bundle from the repository:

```bash
python scripts/package_colab_transfer_v2.py --output results/colab-transfer-v2.zip
```

The packager refuses to overwrite an existing bundle and verifies both original
adapter hashes. No training, optimizer updates, adapter merging, commit or push
is performed by this handoff.
