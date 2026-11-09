"""Check generation-dump compatibility guards without loading a model."""
import ast
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dump_patch', Path(__file__).resolve().parent / 'scripts/patch_generation_dump.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
updated_source = module.updated_source

SOURCE = '''class Trainer:
    def validate(self):
        self._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_extra_infos_dict=extra,
            dump_path=path,
        )
    def rollout(self):
        self._dump_generations(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=extra,
            dump_path=path,
        )
'''


class DumpPatchTests(unittest.TestCase):
    def test_adds_aligned_unknown_ground_truth_at_both_sites(self):
        result = updated_source(SOURCE)
        tree = ast.parse(result)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        for call in calls:
            kwargs = {item.arg: item.value for item in call.keywords}
            value = eval(compile(ast.Expression(kwargs['gts']), '<test>', 'eval'),
                         {'sample_inputs': ['a', 'b'], 'inputs': ['a', 'b']})
            self.assertEqual(value, [None, None])

    def test_idempotent(self):
        result = updated_source(SOURCE)
        self.assertEqual(updated_source(result), result)

    def test_rejects_unknown_call(self):
        with self.assertRaises(ValueError):
            updated_source(SOURCE.replace('inputs=inputs,', 'inputs=unknown,'))

    def test_preserves_explicit_ground_truth(self):
        with self.assertRaises(ValueError):
            updated_source(SOURCE.replace('inputs=inputs,', 'inputs=inputs,\n            gts=labels,'))


if __name__ == '__main__':
    unittest.main()
