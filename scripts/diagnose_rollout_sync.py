"""Compare installed LoRA collectors on a tiny synthetic FSDP model."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import traceback
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/mnt/d/Git/strategic-blocking-llm-agents'))
    parser.add_argument('--ragen-dir', type=Path, default=Path.home() / 'RAGEN')
    args = parser.parse_args()
    project = args.project.resolve()
    ragen = args.ragen_dir.expanduser().resolve()
    results = project / 'results'
    results.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='rollout-sync-audit-', dir=results))
    print(f'Sync audit output: {output}', flush=True)
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import verl.utils.fsdp_utils as fsdp_utils

    installed_utils = Path(inspect.getfile(fsdp_utils)).resolve()
    verl_root = installed_utils.parent.parent
    files = {installed_utils, verl_root/'workers/fsdp_workers.py',
             ragen/'ragen/workers/fsdp_workers.py', ragen/'ragen/trainer/agent_trainer.py'}
    files.update((verl_root/'workers/rollout').rglob('*.py'))
    hashes = {}
    with zipfile.ZipFile(output/'installed-sync-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if not path.is_file():
                raise ValueError(f'Missing installed source: {path}')
            if path.is_relative_to(verl_root):
                name = 'verl/' + path.relative_to(verl_root).as_posix()
            else:
                name = 'ragen/' + path.relative_to(ragen/'ragen').as_posix()
            content = path.read_bytes()
            archive.writestr(name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    report = {'scope': 'Synthetic tiny-model collection checks, not an end-to-end vLLM synchronization test.',
              'source_sha256': hashes, 'cases': [], 'cuda_available': torch.cuda.is_available()}
    if not report['cuda_available']:
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        raise SystemExit('CUDA unavailable; source captured, tensor checks not run.')
    torch.set_num_threads(1)
    dist.init_process_group('nccl', init_method=(output/'distributed-init').as_uri(), rank=0, world_size=1)
    try:
        for layered in (True, False):
            case = {'layered_summon': layered}
            model = None
            try:
                torch.manual_seed(42)
                config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                    use_cache=False)
                config._attn_implementation = 'sdpa'
                base = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
                    r=8, lora_alpha=16, target_modules='all-linear', task_type='CAUSAL_LM'))
                with torch.no_grad():
                    for name, parameter in base.named_parameters():
                        if 'lora_' in name:
                            parameter.fill_(0.125 if 'lora_B' in name else 0.25)
                expected = {name: value.detach().cpu().clone()
                            for name, value in get_peft_model_state_dict(base).items()}
                case['expected_tensor_count'] = len(expected)
                adapter_config = base.peft_config['default'].to_dict()
                case['adapter_settings'] = {key: sorted(value) if isinstance(value, set) else value
                    for key, value in adapter_config.items()
                    if key in ('target_modules', 'r', 'lora_alpha', 'bias', 'lora_dropout')}
                model = FSDP(base, device_id=torch.cuda.current_device(),
                    auto_wrap_policy=fsdp_utils.get_fsdp_wrap_policy(base, is_lora=True),
                    use_orig_params=False)
                collected = fsdp_utils.collect_lora_params(model, layered_summon=layered, base_sync_done=True)
                actual = {name: value.detach().cpu().clone() for name, value in collected.items()}
                case['collected_tensor_count'] = len(actual)
                case['missing_keys'] = sorted(set(expected)-set(actual))
                case['extra_keys'] = sorted(set(actual)-set(expected))
                case['exact_values_match'] = set(actual) == set(expected) and all(
                    torch.equal(actual[name], expected[name]) for name in expected)
                # Emulate changed trainable weights without running an optimizer or a model forward pass.
                with FSDP.summon_full_params(model, writeback=True):
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            if 'lora_B' in name:
                                parameter.fill_(0.375)
                after = fsdp_utils.collect_lora_params(model, layered_summon=layered, base_sync_done=True)
                case['collected_values_change_after_edit'] = any(
                    name in actual and not torch.equal(value.detach().cpu(), actual[name])
                    for name, value in after.items())
            except Exception:
                case['error'] = traceback.format_exc()
            finally:
                if model is not None:
                    del model
                torch.cuda.empty_cache()
            report['cases'].append(case)
            (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({key: value for key, value in case.items()
                              if key not in ('missing_keys', 'extra_keys', 'error', 'adapter_settings')}), flush=True)
            if 'error' in case:
                print('Case failed; traceback saved in summary.json.', flush=True)
    finally:
        dist.destroy_process_group()
    passed = len(report['cases']) == 2 and all(
        case.get('exact_values_match') and case.get('collected_values_change_after_edit')
        for case in report['cases'])
    report['collection_checks_passed'] = passed
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Sync audit completed: {output}', flush=True)
    if not passed:
        raise SystemExit('Collection checks failed. Do not start full training.')
    print('COLLECTION CHECKS PASSED. Real vLLM synchronization remains to be verified.', flush=True)


if __name__ == '__main__':
    main()
