"""Preserve completed updates when validation-based early stopping triggers."""
import ast
from pathlib import Path

OLD = '''                    if self.early_stopped:
                        metrics.update({self._early_stop_metric_key(): 1.0})
                        logger.log(data=metrics, step=self.global_steps)
                        break'''
NEW = '''                    if self.early_stopped:
                        # Preserve the completed update and its advantage metrics before exiting.
                        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                        if self.config.trainer.save_freq > 0:
                            with marked_timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()
                        metrics.update({self._early_stop_metric_key(): 1.0})
                        logger.log(data=metrics, step=self.global_steps)
                        break'''


def updated_source(source):
    if source.count(NEW) == 1 and OLD not in source:
        return source
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError('Unexpected validation early-stop block; inspect before patching.')
    updated = source.replace(OLD, NEW, 1)
    ast.parse(updated)
    return updated


def patch_early_stop_checkpoint(ragen):
    path = Path(ragen) / 'ragen/trainer/agent_trainer.py'
    source = path.read_text(encoding='utf-8')
    updated = updated_source(source)
    if updated == source:
        print('Validation early-stop checkpoint patch already applied.')
        return
    backup = path.with_name('agent_trainer.py.before_early_stop_checkpoint.bak')
    with backup.open('x', encoding='utf-8') as stream:
        stream.write(source)
    path.write_text(updated, encoding='utf-8')
    print(f'Preserved checkpoint and metrics before validation early stop; backup: {backup}')
