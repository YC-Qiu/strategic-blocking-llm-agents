"""Apply guarded compatibility patches for the local RAGEN workflow.

Run this CLI against a separate RAGEN checkout. Each patch checks the source
layout, saves a backup, and skips a change already marked as applied.
"""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ragen-dir", type=Path, required=True)
args = parser.parse_args()
root = args.ragen_dir.expanduser().resolve()
if not (root / "train.py").is_file():
    parser.error(f"Not a RAGEN checkout: {root}")
os.chdir(root)

import ast
from pathlib import Path

actor_path = Path("ragen/workers/actor/dp_actor.py")
source = actor_path.read_text()
original = "from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
marker = "# Load FlashAttention padding helpers only when the unpadded path uses them."
replacement = '''from einops import rearrange

# Load FlashAttention padding helpers only when the unpadded path uses them.
def index_first_axis(*args, **kwargs):
    from flash_attn.bert_padding import index_first_axis as implementation
    return implementation(*args, **kwargs)

def pad_input(*args, **kwargs):
    from flash_attn.bert_padding import pad_input as implementation
    return implementation(*args, **kwargs)

def unpad_input(*args, **kwargs):
    from flash_attn.bert_padding import unpad_input as implementation
    return implementation(*args, **kwargs)
'''
if marker not in source:
    if source.splitlines().count(original) != 1:
        raise SystemExit("Unexpected actor import layout; inspect dp_actor.py before patching.")
    updated = source.replace(original, replacement, 1)
    ast.parse(updated)
    backup = actor_path.with_name("dp_actor.py.before_lazy_padding.bak")
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; inspect it before patching.")
    backup.write_text(source)
    actor_path.write_text(updated)
    print(f"Deferred optional FlashAttention imports; backup: {backup}")
else:
    print("Optional FlashAttention import patch already applied.")

filter_path = Path("ragen/trainer/rollout_filter.py")
source = filter_path.read_text()
marker = "# Align episode reward statistics with every turn before filtering."
if marker not in source:
    old = "reward_std_per_sample = in_group_std.unsqueeze(1).expand(-1, group_size).reshape(-1)"
    anchor = '        metrics["rollout/_selected_group_ids"] = group_ids_for_metrics[top_groups].detach().cpu()'
    if source.count(old) != 2 or source.count(anchor) != 1:
        raise SystemExit("Unexpected reward filter layout; inspect rollout_filter.py before patching.")
    # Compute the aligned tensor once, before either filtering branch.
    updated = "\n".join(line for line in source.split("\n") if line.strip() != old)
    aligned = '''

        # Align episode reward statistics with every turn before filtering.
        if has_episode_ids:
            episode_group_index = {
                eid: index // group_size
                for index, eid in enumerate(unique_episodes)
            }
            turn_group_index = torch.tensor(
                [episode_group_index[eid] for eid in episode_ids],
                dtype=torch.long, device=in_group_std.device,
            )
            reward_std_per_sample = in_group_std[turn_group_index]
        else:
            reward_std_per_sample = in_group_std.repeat_interleave(group_size)
        if reward_std_per_sample.shape[0] != batch.batch.batch_size[0]:
            raise ValueError("Reward statistics do not align with rollout rows.")
'''
    updated = updated.replace(anchor, anchor + aligned, 1)
    ast.parse(updated)
    backup = filter_path.with_name("rollout_filter.py.before_turn_alignment.bak")
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; inspect it before patching.")
    backup.write_text(source)
    filter_path.write_text(updated)
    print(f"Aligned reward statistics with rollout turns; backup: {backup}")
else:
    print("Rollout reward alignment patch already applied.")

worker_path = Path("ragen/workers/fsdp_workers.py")
source = worker_path.read_text()
marker = "# Restore the offload lifecycle omitted by the custom RAGEN update."
if marker not in source:
    original = "        output = self.actor.update_policy(data)"
    if source.splitlines().count(original) != 1:
        raise SystemExit("Unexpected worker update layout; inspect fsdp_workers.py before patching.")
    replacement = '''        # Restore the offload lifecycle omitted by the custom RAGEN update.
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu, load_fsdp_optimizer,
            offload_fsdp_model_to_cpu, offload_fsdp_optimizer,
        )
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        try:
            output = self.actor.update_policy(data)
        finally:
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)'''
    updated = source.replace(original, replacement, 1)
    ast.parse(updated)
    backup = worker_path.with_name("fsdp_workers.py.before_update_offload.bak")
    if backup.exists():
        raise SystemExit(f"Backup already exists: {backup}; inspect it before patching.")
    backup.write_text(source)
    worker_path.write_text(updated)
    print(f"Restored actor update device transfers; backup: {backup}")
else:
    print("Actor update offload patch already applied.")

# This environment has no reference answer text; preserve null ground truths.
from patch_generation_dump import patch_generation_dump
patch_generation_dump(root)

from patch_early_stop_checkpoint import patch_early_stop_checkpoint
patch_early_stop_checkpoint(root)

from patch_lora_sync import patch_lora_sync
patch_lora_sync(root)
