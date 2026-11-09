# Colab transfer evaluation handoff

## What this experiment does

Run inference only for three frozen policies: the original base, the two-step
adapter before blocking practice, and the adapter after blocking practice. Test
all three on the eight frozen maps in `evaluation/frozen-transfer-v1.json`.
Each policy gets 50 episodes per map: 1,200 episodes total, experiment seed
20261112. Preserve failed episodes and all maps; do not tune on the results.

This is a separate Transformers/PEFT protocol, not a reproduction of RAGEN/vLLM.
It uses FP16, SDPA, one sequence at a time, two previous observation/response
pairs, and action-only generation. Malformed or illegal output executes `wait`.
RAGEN can reject malformed output without advancing the environment, so even
the invalid-output handling differs. Never pool these numbers with local RAGEN
results. The purpose is a within-Colab comparison under identical conditions.

The notebook and CPU helpers are prepared and tested locally. A Colab GPU run
has not yet been validated. Treat environment failures as engineering failures,
not evidence about policy performance.

## YC: prepare the handoff

After applying the deliverables patch in Ubuntu:

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -B -m unittest discover -v
python scripts/package_colab_transfer.py --output results/colab-transfer-v1.zip
```

The package includes only the required code, frozen protocol/maps and two verified
adapter exports (roughly 17 MB each before compression). It verifies their hashes
against the frozen suite. No full training checkpoint, optimizer or raw results
are packaged. Keep the ZIP private; do not commit it to Git.

Give your teammate:

1. `notebooks/colab-transfer.ipynb`.
2. `results/colab-transfer-v1.zip`.

This packaging route does not require the latest source changes to be pushed to
GitHub first. The package builder refuses to overwrite an existing ZIP.

## Prakeerth: run in Colab

1. Upload the notebook to Colab and select a GPU runtime.
2. Put the ZIP at `MyDrive/strategic-blocking-transfer/colab-transfer-v1.zip`.
3. Run the notebook cells in order. The install cell pins the non-Torch packages
   to versions recorded in the local training environment. It retains Colab's
   existing CUDA-enabled Torch and records its actual version. These pins are a
   starting configuration, not a guarantee of compatibility on every runtime.
4. Mount Drive, extract into a new temporary folder, and verify every packaged
   file hash. The adapter hashes and frozen map file are checked again by the
   evaluator. Stop on a mismatch.
5. Run the rule-validation cell and then the smoke test on the old one-step
   training fixture. This checks loading/generation for all three policies
   without previewing LLM performance on the new maps.
6. Run the frozen evaluation. Keep the runtime connected and preserve the output
   directory on Drive. The script flushes raw episodes after each completed
   episode, but does not resume partial experiments automatically.
7. Return the entire completed output directory: `manifest.json`,
   `dependencies.txt`, `episodes.jsonl`, `summary.json`, `comparison.md`, and
   `COMPLETE`. A run without `COMPLETE` is partial and must be labeled as such.

If execution stops, send the error and partial run directory. Do not rerun only
unfavorable maps, change the prompt or choose a different seed. Any protocol
change needs a new version and must be reported. A runtime restart can reuse the
Drive ZIP; extraction always creates a new temporary folder.

## Analysis to return

- Report per-map success and Wilson intervals for all three policies.
- Separate symmetry tests, longer corridors, direct-win controls and navigation.
- Report invalid actions and all outcome categories, including timeouts.
- Check whether unnecessary blocking harms direct-win controls.
- Select successful and failed trajectories using a stated rule, not appearance.
- Do not describe action sequences as the model's private thinking process.

The maps were authored after inspecting development results, but frozen before
any LLM evaluation on these maps. This is a small prospective transfer test for
the existing frozen checkpoints, not a broad unbiased benchmark. The next time
these results guide training, the suite becomes development data.

API references used for this implementation:
[PEFT model methods](https://huggingface.co/docs/peft/package_reference/peft_model)
and [Transformers 4.56.1 generation](https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/text_generation).
