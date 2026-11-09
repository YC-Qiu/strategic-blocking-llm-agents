"""Adapt RAGEN generation logging to the installed verl ground-truth argument."""
import ast
from pathlib import Path


def updated_source(source):
    """Patch only the two known call sites; preserve already patched sources."""
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == '_dump_generations']
    expected = {'sample_inputs', 'inputs'}
    edits = []
    seen = set()
    for call in calls:
        arguments = {keyword.arg: keyword.value for keyword in call.keywords}
        inputs = arguments.get('inputs')
        if call.args or not isinstance(inputs, ast.Name) or inputs.id not in expected:
            raise ValueError('Unexpected generation dump call; inspect the installed trainer.')
        if inputs.id in seen:
            raise ValueError('Duplicate generation dump call.')
        seen.add(inputs.id)
        expression = f'[None] * len({inputs.id})'
        if 'gts' in arguments:
            if ast.dump(arguments['gts']) != ast.dump(ast.parse(expression, mode='eval').body):
                raise ValueError('Existing gts argument differs; inspect before modifying.')
            continue
        edits.append((inputs.end_lineno, inputs.col_offset, expression))
    if seen != expected:
        raise ValueError('Expected exactly the validation and rollout dump calls.')
    lines = source.splitlines(keepends=True)
    for line_number, _, expression in sorted(edits, reverse=True):
        line = lines[line_number - 1]
        if not line.strip().startswith('inputs=') or not line.rstrip().endswith(','):
            raise ValueError('Unexpected multiline inputs argument.')
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(line_number, indent + f'gts={expression},\n')
    updated = ''.join(lines)
    ast.parse(updated)
    return updated


def patch_generation_dump(ragen):
    ragen = Path(ragen)
    base = ragen / 'verl/verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(base.read_text(encoding='utf-8'))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == '_dump_generations']
    if len(methods) != 1 or 'gts' not in [arg.arg for arg in methods[0].args.args]:
        raise ValueError('Unsupported verl generation dump signature; inspect before patching.')
    path = ragen / 'ragen/trainer/agent_trainer.py'
    source = path.read_text(encoding='utf-8')
    updated = updated_source(source)
    if updated == source:
        print('Generation dump gts compatibility patch already applied.')
        return
    backup = path.with_name('agent_trainer.py.before_generation_gts.bak')
    with backup.open('x', encoding='utf-8') as stream:
        stream.write(source)
    path.write_text(updated, encoding='utf-8')
    print(f'Patched generation dump gts arguments; backup: {backup}')
