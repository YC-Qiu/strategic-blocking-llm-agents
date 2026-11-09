"""Strict collection for the pilot's single-GPU Qwen2 plain LoRA configuration."""


def _pilot_extract_named_lora(named_parameters, layers, rank):
    import re
    import torch
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    expected = {
        f'base_model.model.model.layers.{layer}.{block}.{target}.lora_{side}.weight'
        for layer in range(layers) for target in targets for side in ('A', 'B')
        for block in ('mlp' if target in {'gate_proj', 'up_proj', 'down_proj'} else 'self_attn',)
    }
    values = {}
    for name, parameter in named_parameters:
        if 'lora_' not in name:
            continue
        name = name.replace('_fsdp_wrapped_module.', '')
        if not re.search(r'\.lora_[AB]\.default\.weight$', name):
            raise ValueError(f'Unsupported pilot LoRA parameter: {name}')
        name = name.replace('.default.weight', '.weight')
        if name not in expected or name in values:
            raise ValueError(f'Unexpected or duplicate pilot LoRA key: {name}')
        value = parameter.detach().to('cpu').contiguous().clone()
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid pilot LoRA tensor: {name}')
        if value.shape[0 if '.lora_A.' in name else 1] != rank:
            raise ValueError(f'Unexpected pilot LoRA rank: {name}')
        values[name] = value
    if set(values) != expected:
        raise ValueError(f'Incomplete pilot LoRA collection: expected {len(expected)}, got {len(values)}; '
                         f'missing={sorted(expected-set(values))[:3]}')
    return values


def _pilot_collect_lora_params(module):
    import hashlib
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if not isinstance(module, FSDP) or not dist.is_initialized() or dist.get_world_size() != 1:
        raise ValueError('Pilot LoRA collection requires FSDP1 and distributed world size one.')
    model = module._fsdp_wrapped_module
    configs = getattr(model, 'peft_config', {})
    if set(configs) != {'default'}:
        raise ValueError('Pilot collector requires exactly one default adapter.')
    config = configs['default'].to_dict()
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    raw_targets = config.get('target_modules')
    if isinstance(raw_targets, str):
        targets_match = raw_targets == 'all-linear'
    else:
        targets_match = {str(name).rsplit('.', 1)[-1] for name in (raw_targets or [])} == targets
    # Exact layer/projection coverage is checked separately against the extracted tensors.
    if (not targets_match or config.get('r') != 8
            or config.get('lora_alpha') != 16 or config.get('bias') != 'none'
            or config.get('lora_dropout') != 0):
        details = {key: config.get(key) for key in
                   ('target_modules', 'r', 'lora_alpha', 'bias', 'lora_dropout')}
        raise ValueError(f'Unsupported pilot adapter configuration: {details!r}')
    for option in ('use_dora', 'use_rslora', 'fan_in_fan_out', 'lora_bias', 'modules_to_save',
                   'rank_pattern', 'alpha_pattern', 'layer_replication', 'target_parameters',
                   'trainable_token_indices', 'use_qalora', 'alora_invocation_tokens', 'use_bdlora'):
        if config.get(option):
            raise ValueError(f'Unsupported pilot adapter option: {option}')
    if getattr(model.config, 'model_type', None) != 'qwen2':
        raise ValueError('Pilot collector only supports the verified Qwen2 model layout.')
    # Recursively expose unflattened parameters; copy only adapters before the context exits.
    with FSDP.summon_full_params(module, recurse=True, writeback=False):
        values = _pilot_extract_named_lora(module.named_parameters(), model.config.num_hidden_layers, 8)
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    print(f'PILOT_LORA_COLLECT tensors={len(values)} sha256={digest.hexdigest()}', flush=True)
    return values
