"""Export numeric console metrics and unsmoothed pilot learning curves."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

STEP = re.compile(r'\bstep:(\d+)\s+-\s+(.*)')
NUMBER = re.compile(r'^(.+):([+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan|inf))$')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse_metrics(text):
    rows = {}
    for line in ANSI.sub('', text).splitlines():
        match = STEP.search(line)
        if not match:
            continue
        row = rows.setdefault(int(match[1]), {'step': int(match[1])})
        for field in match[2].split(' - '):
            metric = NUMBER.fullmatch(field.strip())
            if metric:
                value = float(metric[2])
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite metric at step {match[1]}: {metric[1]}')
                row[metric[1]] = value
    if not rows:
        raise ValueError('No numeric step metrics found in this log.')
    return [rows[step] for step in sorted(rows)]


def summarize(run_dir):
    rows = parse_metrics((run_dir / 'run.log').read_text(encoding='utf-8', errors='replace'))
    train_rows = [row for row in rows if 'actor/pg_loss' in row]
    if not train_rows:
        raise ValueError('No actor training metrics found.')
    keys = ['step'] + sorted({key for row in rows for key in row if key != 'step'})
    with (run_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'scope': 'Training rollouts are sampled before each update; greedy validation occurs separately.',
        'recorded_training_steps': [row['step'] for row in train_rows],
        'training_steps_missing_advantage_metrics': [row['step'] for row in train_rows
            if 'critic/advantages/min' not in row or 'critic/advantages/max' not in row],
        'early_stop_observed': any(key.startswith('early_stopped/') and value != 0
            for row in rows for key, value in row.items()),
        'saved_actor_checkpoints': [str(path.relative_to(run_dir))
            for path in sorted((run_dir / 'checkpoints').glob('global_step_*/actor/model*.pt'))
            if path.stat().st_size > 0],
        'training_steps_with_nonzero_advantage': sum(
            row.get('critic/advantages/min', 0) != 0 or row.get('critic/advantages/max', 0) != 0
            for row in train_rows),
        'learning_demonstrated': 'Requires matched before/after evaluation; not inferred automatically.',
        'rows': rows,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [
        ('Episode return', [('train/charts/avg_episode_return', 'Training samples'),
                            ('val-env/BlockingWarmupRight/episodic_return', 'Greedy: right'),
                            ('val-env/BlockingWarmupLeft/episodic_return', 'Greedy: left')]),
        ('Policy-gradient loss', [('actor/pg_loss', 'Actor loss')]),
        ('Success fraction', [('train/BlockingWarmupRight/success', 'Training: right'),
                              ('train/BlockingWarmupLeft/success', 'Training: left'),
                              ('val-env/BlockingWarmupRight/success', 'Greedy: right'),
                              ('val-env/BlockingWarmupLeft/success', 'Greedy: left')]),
        ('Advantage range', [('critic/advantages/min', 'Minimum'), ('critic/advantages/max', 'Maximum')]),
    ]
    for axis, (title, series) in zip(axes.flat, panels):
        for key, label in series:
            points = [(row['step'], row[key]) for row in rows if key in row]
            if points:
                axis.plot(*zip(*points), marker='o', linewidth=1.4, label=label)
        axis.set_title(title)
        axis.set_xlabel('Training iteration (0 = before training)')
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.suptitle('Navigation warmup pilot: observed metrics, no smoothing')
    fig.savefig(run_dir / 'learning-curves.png', dpi=160)
    fig.savefig(run_dir / 'learning-curves.pdf')
    plt.close(fig)
    print(f"Recorded training iterations: {result['recorded_training_steps']}")
    print(f"Iterations with nonzero advantages: {result['training_steps_with_nonzero_advantage']}")
    if result['training_steps_missing_advantage_metrics']:
        print(f"Advantage metrics MISSING (not zero): {result['training_steps_missing_advantage_metrics']}")
    if result['early_stop_observed']:
        print('WARNING: trainer reported early stopping; recorded steps do not imply normal completion.')
    if not result['saved_actor_checkpoints']:
        print('WARNING: no saved actor checkpoint found in this run.')
    for row in train_rows:
        rates = [f"{name}={row[key]:.1%}" for name, key in (
            ("right", "train/BlockingWarmupRight/success"),
            ("left", "train/BlockingWarmupLeft/success"),
        ) if key in row]
        if rates:
            print(f"Training success at step {row['step']}: " + ", ".join(rates))
    print(f'Curves and metrics: {run_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    summarize(parser.parse_args().run_dir.resolve())
