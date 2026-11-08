"""Check runtime imports, LoRA updates, and RAGEN FSDP offload transitions.

Run from the configured RAGEN checkout with the generated YAML path as argv[1].
This is a small-model integration check, not a full training guarantee.
"""

import contextlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import yaml

for package in ("torch", "transformers", "peft", "torchao", "vllm", "tensordict"):
    try:
        print(f"{package}: {version(package)}", flush=True)
    except PackageNotFoundError:
        print(f"{package}: not installed", flush=True)

import torch
import vllm
import verl
from peft import LoraConfig, get_peft_model
from peft.import_utils import is_torchao_available
from transformers import Qwen2Config, Qwen2ForCausalLM

# Ordinary LoRA does not require torchao, but stale installations can break PEFT.
try:
    is_torchao_available()
except ImportError as exc:
    raise SystemExit(
        f"PEFT optional dependency check failed: {exc}\n"
        "This run does not use torchao quantization. In this environment, run:\n"
        "python -m pip uninstall -y torchao\n"
        "Then rerun this script."
    ) from exc

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
assert torch.cuda.is_available(), "CUDA is unavailable."
torch.set_num_threads(1)
torch.manual_seed(42)
for device in ("cpu", "cuda"):
    print(f"Checking tiny Qwen2 LoRA forward/backward/update on {device}...", flush=True)
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=64,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)
    config._attn_implementation = "sdpa"
    base = Qwen2ForCausalLM(config)
    model = get_peft_model(base, LoraConfig(
        r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM"))
    model.to(device).train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    assert trainable and frozen, "Expected trainable adapters and frozen base weights."
    assert all("lora_" in n for n in trainable), "Unexpected trainable base weights."
    before = {n: p.detach().clone() for n, p in trainable.items()}
    frozen_before = {n: p.detach().clone() for n, p in frozen.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=cfg["actor_rollout_ref"]["actor"]["optim"]["lr"])
    tokens = torch.randint(3, 128, (1, 16), device=device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    with amp:
        loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=tokens).loss
    assert torch.isfinite(loss), "Non-finite smoke-test loss."
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in trainable.values()), "Missing adapter gradients."
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values()), "Non-finite adapter gradients."
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in trainable.items()), "Adapters did not update."
    assert all(torch.equal(frozen_before[n], p) for n, p in frozen.items()), "Frozen weights changed."
    print(f"PASS: {device} LoRA update; loss={loss.item():.6f}", flush=True)
    del optimizer, model, base, trainable, frozen, before, frozen_before
# Exercise the installed worker's device-transfer method without starting Ray.
import ast
import tempfile
from types import SimpleNamespace
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_model_to_cpu
from ragen.workers.actor.dp_actor import DataParallelPPOActor
import ragen.workers.fsdp_workers as worker_module

tree = ast.parse(Path(worker_module.__file__).read_text())
worker_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker")
update_method = next(n for n in worker_class.body if isinstance(n, ast.FunctionDef) and n.name == "update_actor")
update_method.decorator_list = []
namespace = dict(vars(worker_module))
exec(compile(ast.fix_missing_locations(ast.Module(body=[update_method], type_ignores=[])),
             worker_module.__file__, "exec"), namespace)
worker_update = namespace["update_actor"]
print("Checking installed RAGEN worker: two FSDP LoRA updates across CPU/GPU offload...", flush=True)
with tempfile.TemporaryDirectory(dir=Path(sys.argv[1]).resolve().parent) as scratch:
    rendezvous = (Path(scratch) / "distributed-init").as_uri()
    dist.init_process_group("nccl", init_method=rendezvous, rank=0, world_size=1)
    try:
        base = Qwen2ForCausalLM(config)
        model = get_peft_model(base, LoraConfig(
            r=cfg["lora"]["rank"], lora_alpha=cfg["lora"]["alpha"],
            target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM"))
        model = FSDP(model, device_id=torch.cuda.current_device(),
                     auto_wrap_policy=get_fsdp_wrap_policy(model, is_lora=True), use_orig_params=False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        actor = SimpleNamespace(actor_module=model, actor_optimizer=optimizer,
                                config=SimpleNamespace(grad_clip=1.0))

        def update_policy(data):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=data.batch["input_ids"], labels=data.batch["input_ids"],
                             attention_mask=data.batch["attention_mask"]).loss
            assert torch.isfinite(loss), "Non-finite FSDP test loss."
            loss.backward()
            grad_norm = DataParallelPPOActor._optimizer_step(actor)
            assert torch.isfinite(grad_norm) and grad_norm > 0
            optimizer.zero_grad(set_to_none=True)
            return {"smoke_loss": loss.item()}

        worker = SimpleNamespace(actor=SimpleNamespace(update_policy=update_policy),
                                 actor_module_fsdp=model, actor_optimizer=optimizer,
                                 _is_offload_param=True, _is_offload_optimizer=True)
        offload_fsdp_model_to_cpu(model)
        for iteration in range(2):
            before = [p.detach().cpu().clone() for p in model.parameters()]
            data = DataProto.from_dict(tensors={
                "input_ids": torch.randint(3, 128, (1, 16)),
                "attention_mask": torch.ones((1, 16), dtype=torch.long)})
            worker_update(worker, data)
            torch.cuda.synchronize()
            assert any(not torch.equal(old, p.detach().cpu()) for old, p in zip(before, model.parameters()))
            assert all(p.device.type == "cpu" for p in model.parameters()), "Model offload failed."
            assert optimizer.state, "Optimizer state was not initialized."
            assert all(v.device.type == "cpu" for state in optimizer.state.values()
                       for v in state.values() if torch.is_tensor(v)), "Optimizer offload failed."
            print(f"PASS: FSDP LoRA offload/update cycle {iteration + 1}", flush=True)
        del model, optimizer, actor, worker
    finally:
        dist.destroy_process_group()
print("PREFLIGHT PASSED: tiny LoRA and installed worker offload updates; full Ray rollout training remains to be tested.", flush=True)
