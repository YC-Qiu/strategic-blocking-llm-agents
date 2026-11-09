"""Guarded repair of the captured pilot collector and synchronous vLLM adapter refresh."""
import ast
import hashlib
from pathlib import Path

# Populated from the captured installed sources when this patch is packaged.
EXPECTED = {'verl/verl/utils/fsdp_utils.py': {'before': '57069140830aad0b43c9576df666d28b494683b821f019cfd592598f759444ce', 'after': '4e67f67d1b88a85d8cd0be8662dba5a56e35486d047658ad01816bf508aa6b90', 'previous': '5dfbe85502212dd033ffee5f110bc058ee78b3353073f0bf826730f22ade1b27'}, 'verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py': {'before': 'f5b405f5b0824328d95cdbb089c2e0b6b6e5f719cf3714dd55f01c57b1161b0f', 'after': '11fc3594b34be2a8a650668fa842c8c9126691700b8313c4d8fde4136144fba2'}}
OLD_TARGET_CHECK = "    if (set(config.get('target_modules', [])) != targets or config.get('r') != 8\n            or config.get('lora_alpha') != 16 or config.get('bias') != 'none'\n            or config.get('lora_dropout') != 0):\n        raise ValueError('Unsupported pilot adapter configuration.')"
NEW_TARGET_CHECK = "    raw_targets = config.get('target_modules')\n    if isinstance(raw_targets, str):\n        targets_match = raw_targets == 'all-linear'\n    else:\n        targets_match = {str(name).rsplit('.', 1)[-1] for name in (raw_targets or [])} == targets\n    # Exact layer/projection coverage is checked separately against the extracted tensors.\n    if (not targets_match or config.get('r') != 8\n            or config.get('lora_alpha') != 16 or config.get('bias') != 'none'\n            or config.get('lora_dropout') != 0):\n        details = {key: config.get(key) for key in\n                   ('target_modules', 'r', 'lora_alpha', 'bias', 'lora_dropout')}\n        raise ValueError(f'Unsupported pilot adapter configuration: {details!r}')"

OLD_SYNC = '''            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")'''
NEW_SYNC = '''            # Pilot: refuse empty adapters and replace the prior generation adapter.
            pairs = list(weights)
            tensors = dict(pairs)
            if not tensors or len(tensors) != len(pairs):
                raise ValueError("Empty or duplicate LoRA tensors supplied to vLLM")
            engine = self.inference_engine.llm_engine
            for old_id in list(engine.list_loras()):
                engine.remove_lora(old_id)
            lora_int_id = VLLM_LORA_INT_ID
            request = TensorLoRARequest(
                lora_name=f"{lora_int_id}", lora_int_id=lora_int_id,
                lora_path="simon_lora_path", peft_config=asdict(peft_config),
                lora_tensors=tensors,
            )
            engine.add_lora(request)
            if set(engine.list_loras()) != {lora_int_id}:
                raise RuntimeError("vLLM did not retain exactly the current pilot adapter")
            print(f"PILOT_LORA_SYNC tensors={len(tensors)} active_id={lora_int_id}", flush=True)'''


def collector_source(source, helpers):
    tree = ast.parse(source)
    layered = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'layered_summon_lora_params')
    collect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_lora_params')
    if not isinstance(collect.body[0], ast.Expr) or not isinstance(collect.body[0].value, ast.Constant):
        raise ValueError('Unexpected collector docstring layout.')
    lines = source.splitlines(keepends=True)
    # Apply the later edit first to preserve original line positions.
    insertion = collect.body[0].end_lineno
    lines.insert(insertion, '    if base_sync_done:\n        return _pilot_collect_lora_params(module)\n')
    lines[layered.lineno-1:layered.end_lineno] = [helpers.rstrip() + '\n\n\n',
        'def layered_summon_lora_params(fsdp_module) -> OrderedDict:\n',
        '    return _pilot_collect_lora_params(fsdp_module)\n']
    updated = ''.join(lines)
    ast.parse(updated)
    return updated


def rollout_source(source):
    if source.count(OLD_SYNC) != 1:
        raise ValueError('Unexpected synchronous LoRA update implementation.')
    updated = source.replace(OLD_SYNC, NEW_SYNC, 1)
    ast.parse(updated)
    return updated


def patch_lora_sync(ragen):
    root = Path(ragen)
    helpers = Path(__file__).with_name('pilot_lora_collection.py').read_text(encoding='utf-8')
    plans = []
    for relative, hashes in EXPECTED.items():
        path = root / relative
        source = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        if digest == hashes['after']:
            print(f'Pilot LoRA sync patch already applied: {relative}')
            continue
        migrating_targets = digest == hashes.get('previous')
        if migrating_targets:
            if source.count(OLD_TARGET_CHECK) != 1:
                raise ValueError('Unexpected previous pilot target check.')
            updated = source.replace(OLD_TARGET_CHECK, NEW_TARGET_CHECK, 1)
        elif digest == hashes['before']:
            updated = collector_source(source, helpers) if relative.endswith('fsdp_utils.py') else rollout_source(source)
        else:
            raise ValueError(f'Installed source changed; inspect before patching: {path}')
        if hashlib.sha256(updated.encode('utf-8')).hexdigest() != hashes['after']:
            raise ValueError('Patch payload changed; regenerate and review the compatibility patch.')
        suffix = '.before_pilot_lora_target_check.bak' if migrating_targets else '.before_pilot_lora_sync.bak'
        backup = path.with_name(path.name + suffix)
        if backup.exists():
            raise ValueError(f'Backup already exists; inspect before patching: {backup}')
        plans.append((path, backup, source, updated))
    if not EXPECTED:
        raise ValueError('Missing captured-source hashes.')
    for path, backup, source, updated in plans:
        with backup.open('x', encoding='utf-8') as stream:
            stream.write(source)
        path.write_text(updated, encoding='utf-8')
        print(f'Applied pilot LoRA sync repair: {path}')
