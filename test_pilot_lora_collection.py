"""Check pilot adapter extraction without FSDP or a GPU."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('pilot_collection', Path(__file__).resolve().parent/'scripts/pilot_lora_collection.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
extract = module._pilot_extract_named_lora


def parameters():
    values = []
    for target in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        block = 'mlp' if target in ('gate_proj', 'up_proj', 'down_proj') else 'self_attn'
        for side in ('A', 'B'):
            name = f'_fsdp_wrapped_module.base_model.model.model.layers.0._fsdp_wrapped_module.{block}.{target}.lora_{side}.default.weight'
            values.append((name, torch.ones((8, 4) if side == 'A' else (4, 8))))
    return values


class CollectionTests(unittest.TestCase):
    def test_complete_named_parameters_and_independent_cpu_copy(self):
        original = parameters()
        output = extract(iter(original), 1, 8)
        self.assertEqual(len(output), 14)
        self.assertTrue(all('_fsdp_wrapped_module' not in key and '.default.' not in key for key in output))
        original[0][1].zero_()
        self.assertTrue(all(torch.all(value == 1) for value in output.values()))

    def test_empty_and_incomplete_are_rejected(self):
        for values in ([], parameters()[:-1]):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                extract(values, 1, 8)

    def test_duplicate_is_rejected(self):
        values = parameters()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            extract(values + values[:1], 1, 8)

    def test_nonfinite_is_rejected(self):
        values = parameters()
        values[0][1][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            extract(values, 1, 8)

    def test_unexpected_adapter_is_rejected(self):
        values = [(name.replace('.default.', '.other.'), tensor) for name, tensor in parameters()]
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            extract(values, 1, 8)

    def test_rank_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'rank'):
            extract(parameters(), 1, 4)


if __name__ == '__main__':
    unittest.main()
