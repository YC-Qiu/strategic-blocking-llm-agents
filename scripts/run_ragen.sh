#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PROJECT="${PROJECT:-$(cd -- "$script_dir/.." && pwd)}"
RAGEN_DIR="${RAGEN_DIR:-$HOME/RAGEN}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$RAGEN_DIR"

mode=${1:-slides}
case "$mode" in
    slides|costly) stage=before_rl ;;
    train-one|train-warmup|train-five|train-sync|train-balanced) stage=train ;;
    *) echo "Usage: bash scripts/run_ragen.sh [slides|costly|train-one|train-warmup|train-five|train-sync|train-balanced] [--check-only]" >&2; exit 2 ;;
esac
mkdir -p "$PROJECT/results/$stage"
run_dir=$(mktemp -d "$PROJECT/results/$stage/$mode-XXXXXXXX")
config_file=$(mktemp config/strategic_blocking_actions_XXXXXXXX.yaml)
python - "$config_file" "$run_dir" "$mode" <<'PY'
import os
import sys
from pathlib import Path
import yaml

path, out = map(Path, sys.argv[1:3])
mode = sys.argv[3]
cfg = yaml.safe_load((Path(os.environ["PROJECT"]) / "strategic_blocking.yaml").read_text())
cfg["model_path"] = "Qwen/Qwen2.5-0.5B-Instruct"
cfg["agent_proxy"]["enable_think"] = False
instruction = cfg["custom_envs"]["BlockingSlides"]["env_instruction"] + (
    "\nYou choose A's action only. B has already moved. Copy one exact string "
    "from Legal actions, including its direction if present. Return it inside "
    "<answer>...</answer>. Use words, never an option number. Do not add an explanation."
)
for tag in ("BlockingSlides", "BlockingCostly"):
    cfg["custom_envs"][tag]["env_instruction"] = instruction
val_tag = "BlockingCostly" if mode == "costly" else "BlockingSlides"
cfg["es_manager"]["val"].update(env_groups=1, group_size=1,
    env_configs=dict(tags=[val_tag], n_groups=[1]))
cfg["actor_rollout_ref"] = dict(rollout=dict(tensor_model_parallel_size=1,
    max_model_len=2048, response_length=128, gpu_memory_utilization=0.7,
    val_kwargs=dict(do_sample=False, temperature=0)))
cfg["output"].update(dir=str(out), filename="rollout.pkl", format="pkl",
                     append_timestamp=False, save_pkl_backup=False, save_jsonl_backup=False)
if mode in ("train-one", "train-warmup", "train-five", "train-sync", "train-balanced"):
    cfg["system"] = {"CUDA_VISIBLE_DEVICES": "0"}
    # Train adapters to limit gradient and optimizer memory on the local machine.
    cfg["lora"] = dict(rank=8, alpha=16, target_modules="all-linear")
    cfg["ppo_mini_batch_size"] = 2
    cfg["micro_batch_size_per_gpu"] = 1
    cfg["es_manager"]["train"].update(env_groups=2, group_size=2,
        env_configs=dict(tags=["BlockingSlides", "BlockingCostly"], n_groups=[1, 1]))
    cfg["algorithm"] = dict(adv_estimator="grpo", bi_level_gae=False,
                            norm_adv_by_std_in_grpo=True, use_kl_in_reward=False)
    cfg["actor_rollout_ref"]["actor"] = dict(use_ref=False, use_kl_loss=False,
        loss_agg_mode="seq-mean-token-mean", ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1, use_torch_compile=False,
        optim=dict(lr=1e-4),
        fsdp_config=dict(param_offload=True, optimizer_offload=True))
    # Use PyTorch attention for the trainer; the installed FlashAttention wheel
    # requires a newer glibc than this WSL distribution provides.
    cfg["actor_rollout_ref"]["model"] = dict(enable_gradient_checkpointing=True,
        override_config=dict(attn_implementation="sdpa"), use_remove_padding=False)
    # Limit inference profiling to this four-trajectory smoke test and leave
    # enough of the engine's memory budget for weights, activations, and KV cache.
    cfg["actor_rollout_ref"]["rollout"].update(gpu_memory_utilization=0.50,
        max_num_seqs=4, max_num_batched_tokens=2048, enforce_eager=True,
        load_format="safetensors", layered_summon=True,
        temperature=1.0, do_sample=True, log_prob_micro_batch_size_per_gpu=1,
        rollout_filter_value=1.0, rollout_filter_include_zero=True)
    cfg["trainer"].update(total_training_steps=1, val_before_train=False, test_freq=-1,
        save_freq=1, logger=["console"], n_gpus_per_node=1, nnodes=1,
        default_local_dir=str(out/"checkpoints"))
if mode in ("train-warmup", "train-five", "train-sync", "train-balanced"):
    from copy import deepcopy
    warmup = deepcopy(cfg["custom_envs"]["BlockingSlides"])
    warmup["env_config"]["layout"] = "warmup_right"
    cfg["custom_envs"]["BlockingWarmupRight"] = warmup
    # Eight independent samples from one initial state; four concurrent sequences.
    cfg["es_manager"]["train"].update(env_groups=1, group_size=8,
        env_configs=dict(tags=["BlockingWarmupRight"], n_groups=[1]))
    cfg["es_manager"]["val"].update(env_groups=1, group_size=1,
        env_configs=dict(tags=["BlockingWarmupRight"], n_groups=[1]))
    cfg["trainer"]["experiment_name"] = "strategic-blocking-warmup-diagnostic"
    # Isolate the reward-driven objective in this diagnostic iteration.
    cfg["actor_rollout_ref"]["actor"]["entropy_coeff"] = 0.0
    cfg["actor_rollout_ref"]["actor"]["optim"]["weight_decay"] = 0.0
if mode in ("train-five", "train-sync", "train-balanced"):
    # Fresh continuous pilot, not five independent one-step runs.
    cfg["trainer"].update(experiment_name="strategic-blocking-five-step-pilot",
        total_training_steps=5, val_before_train=True, test_freq=1,
        validation_steps=1, save_freq=5,
        validation_data_dir=str(out / "validation"))
    cfg["actor_rollout_ref"]["actor"]["optim"]["lr"] = 2e-5
    # One deterministic validation episode per map at baseline and after updates.
    left = deepcopy(cfg["custom_envs"]["BlockingWarmupRight"])
    left["env_config"]["layout"] = "warmup_left"
    cfg["custom_envs"]["BlockingWarmupLeft"] = left
    cfg["es_manager"]["val"].update(env_groups=2, group_size=1,
        env_configs=dict(tags=["BlockingWarmupRight", "BlockingWarmupLeft"], n_groups=[1, 1]))
if mode == "train-sync":
    # Check collection, live adapter registration, and two successive updates.
    cfg["trainer"].update(experiment_name="strategic-blocking-sync-check",
        total_training_steps=2, save_freq=2)
if mode == "train-balanced":
    # Start from the base model and sample each direction as a separate GRPO group.
    # Keep eight samples per group; this doubles the total rollout budget per update.
    cfg["trainer"]["experiment_name"] = "strategic-blocking-balanced-five-step-pilot"
    cfg["es_manager"]["train"].update(env_groups=2, group_size=8,
        env_configs=dict(tags=["BlockingWarmupRight", "BlockingWarmupLeft"], n_groups=[1, 1]))
text = yaml.safe_dump(cfg, sort_keys=False)
path.write_text(text)
(out/"config.yaml").write_text(text)
PY

printf 'Results: %s\n' "$run_dir"
if [[ "$mode" == train-one || "$mode" == train-warmup || "$mode" == train-five || "$mode" == train-sync || "$mode" == train-balanced ]]; then
    python "$PROJECT/scripts/patch_ragen.py" --ragen-dir "$RAGEN_DIR"
    # Fail early on actor import errors before allocating models and KV cache.
    python -c 'from ragen.workers.actor.dp_actor import DataParallelPPOActor; print("RAGEN actor import passed")'
    python -m pip freeze > "$run_dir/dependencies.txt"
    if ! python -m pip check > "$run_dir/pip-check.txt" 2>&1; then
        printf 'Dependency metadata conflicts recorded in %s/pip-check.txt\n' "$run_dir"
    fi
    python -u "$PROJECT/scripts/check_ragen.py" "$config_file" 2>&1 | tee "$run_dir/preflight.log"
    name=$(basename "$config_file" .yaml)
    python train.py --config-name "$name" --cfg job > "$run_dir/composed.yaml"
    if [[ "${2:-}" == --check-only ]]; then
        printf 'Preflight log: %s/preflight.log\n' "$run_dir"
        exit 0
    fi
    python -u train.py --config-name "$name" 2>&1 | tee "$run_dir/run.log"
    if [[ "$mode" == train-five || "$mode" == train-sync || "$mode" == train-balanced ]]; then
        python "$PROJECT/scripts/summarize_training.py" "$run_dir"
    fi
    printf 'Training log: %s/run.log\n' "$run_dir"
    exit 0
fi
python -u -m ragen.llm_agent.agent_proxy --config-path "$PWD/config" \
    --config-name "$(basename "$config_file" .yaml)" 2>&1 | tee "$run_dir/run.log"

python - "$run_dir" <<'PY'
import json, pickle, sys
from pathlib import Path

out = Path(sys.argv[1])
with (out / "rollout.pkl").open("rb") as f:
    data = pickle.load(f)  # Local artifact produced by the invocation above.
batch = data.non_tensor_batch
assert len(batch["history"]) == 1
history = list(batch["history"][0])
def encode(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(type(obj).__name__)
record = dict(history=history, messages_list=batch["messages_list"], metrics=batch["metrics"][0])
(out/"episode.jsonl").write_text(json.dumps(record, default=encode) + "\n")
print("Executed actions:", [a for row in history for a in row.get("actions", [])])
print("Final observation:", history[-1]["state"])
print("Saved:", out/"episode.jsonl")
PY
