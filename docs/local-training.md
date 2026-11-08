# Local evaluation and training workflow

This workflow packages the launcher and compatibility fixes used for the first completed local training smoke test. It is intended for an existing RAGEN installation, not as a validated clean-install recipe.

## Prerequisites

- A separate RAGEN checkout, normally at `$HOME/RAGEN`.
- A working Linux CUDA environment. The recorded run used Ubuntu under WSL2 and an RTX 3070 with 8 GB VRAM.
- The maze registered in RAGEN as described in [RAGEN.md](../RAGEN.md).
- The project's native tests passing with `python -m unittest -v`.

Selected versions recorded with the successful run were Python 3.12, PyTorch 2.8.0, Transformers 4.56.1, PEFT 0.21.2, vLLM 0.11.0, and TensorDict 0.10.0. `torchao` was absent. These are observations from the working environment, not a mutually compatible dependency lock: `pip check` still reported conflicts, including RAGEN's declared vLLM and TensorDict requirements.

The local system used SDPA because its installed FlashAttention extension required a newer glibc. Ordinary LoRA did not require `torchao`; removing its incompatible optional installation resolved PEFT adapter creation. Do not blindly upgrade the entire dependency stack to reproduce this run.

## Run from the project checkout

Activate the existing environment and enter this repository in Bash:

```bash
conda activate ragen
cd /path/to/strategic-blocking-llm-agents
export RAGEN_DIR="$HOME/RAGEN"
```

The launcher derives `PROJECT` from its own location unless that environment variable is already set. Override `RAGEN_DIR` if the external checkout is elsewhere. The launcher adds the project to `PYTHONPATH` and generates a temporary configuration inside RAGEN's `config/` directory.

Before-RL evaluations, one fixed layout per command:

```bash
bash scripts/run_ragen.sh slides
bash scripts/run_ragen.sh costly
```

Each evaluation saves its configuration, log, local rollout pickle, and an `episode.jsonl` containing the trajectory and model messages. The exporter reads only the pickle created by that invocation; do not substitute an untrusted pickle.

Run training preflight without launching the full training job:

```bash
bash scripts/run_ragen.sh train-one --check-only
```

Run one training iteration:

```bash
bash scripts/run_ragen.sh train-one
```

Each command creates a separate directory under `results/before_rl/` or `results/train/`. A training iteration can contain multiple optimizer steps because the multi-turn episodes are expanded into individual training records. `train-one` starts a new run; it does not resume the previous checkpoint.

## Included compatibility changes

Before training preflight, `scripts/patch_ragen.py` modifies three files in the external RAGEN checkout:

| RAGEN file | Change |
| --- | --- |
| `ragen/workers/actor/dp_actor.py` | Delay optional FlashAttention padding imports until the unpadded path actually uses them; import `rearrange` from `einops`. |
| `ragen/trainer/rollout_filter.py` | Expand episode-level reward statistics onto the corresponding turn-level rows before keeping or filtering trajectories. |
| `ragen/workers/fsdp_workers.py` | Restore parameter and optimizer loading before an actor update, and CPU offload afterward. |

Each patch requires an expected source pattern, preserves a neighboring `.bak` file, and recognizes its own marker on subsequent runs. An unexpected layout stops the script instead of attempting a broad replacement. The patches were derived from the observed local failures and inspected upstream code; no exact tested RAGEN commit hash is available yet.

## What preflight verifies

`scripts/check_ragen.py` runs before full model loading and Ray training:

1. Report key package versions and import the runtime dependencies.
2. Create a tiny Qwen2 model with the configured LoRA rank and target modules.
3. Perform CPU and GPU forward, backward, and optimizer steps; check finite gradients, changed adapter weights, and frozen base weights.
4. Use the installed RAGEN worker's `update_actor` method without its Ray decorators for two small FSDP LoRA update cycles, checking CPU/GPU transfers of both parameters and initialized optimizer state.

This is a synthetic integration check. It does not execute the full maze reward objective, Ray scheduling, vLLM weight synchronization, or checkpoint reload. Passing preflight does not guarantee a full training run will succeed.

The launcher stores `dependencies.txt`, `pip-check.txt`, `preflight.log`, generated `config.yaml`, and Hydra's `composed.yaml`. Dependency metadata conflicts are recorded but do not automatically stop the synthetic runtime tests. Runtime failures stop the launcher before full training.

## Recorded outcome and remaining work

Run `train-one-F5JCCw3d` completed one iteration on November 7, 2026. Its four episodes all lost, so every GRPO advantage was zero. Adapter parameters changed, primarily from entropy regularization; strategic learning was not established. Curated numeric results are in [smoke-test-2026-11-07.json](smoke-test-2026-11-07.json).

The main FSDP checkpoint retains 336 finite LoRA tensors, but the standalone adapter export is empty and must not be used for evaluation. Adapter export/reload, multi-iteration stability, useful reward variation, post-training evaluation, and held-out maze evaluation remain open.

The packaged scripts preserve the previously exercised logic with portable paths. Local syntax, configuration-generation, patch, and native-environment checks can be run without the GPU stack; the refactored entry point itself still needs a full WSL/GPU rerun. Raw results and model weights remain local and are excluded from Git. Keep only curated summaries and reviewed figures under `docs/`.
