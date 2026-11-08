"""Check recovery guards without loading a large model or requiring PEFT."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from export_pilot_adapter import extract_adapter, TARGETS


class ExportGuards(unittest.TestCase):
    def setUp(self):
        self.adapter = {
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none",
            "task_type": "CAUSAL_LM", "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "target_modules": sorted(TARGETS), "init_lora_weights": True,
        }
        self.model = {"model_type": "qwen2", "num_hidden_layers": 1}
        self.fsdp = {"world_size": 1, "FSDP_version": 1}
        self.state = {}
        for target in TARGETS:
            block = "mlp" if target in {"gate_proj", "up_proj", "down_proj"} else "self_attn"
            prefix = f"base_model.model.model.layers.0.{block}.{target}"
            self.state[prefix + ".base_layer.weight"] = torch.ones(3, 2)
            self.state[prefix + ".lora_A.default.weight"] = torch.ones(8, 2)
            self.state[prefix + ".lora_B.default.weight"] = torch.ones(3, 8)

    def extract(self):
        return extract_adapter(self.state, self.adapter, self.model, self.fsdp)

    def test_complete_adapter_and_normalized_keys(self):
        tensors, config, nonzero = self.extract()
        self.assertEqual(len(tensors), 14)
        self.assertEqual(nonzero, 7)
        self.assertTrue(config["inference_mode"])
        self.assertTrue(all(".default." not in key for key in tensors))
        before = next(value.clone() for key, value in self.state.items() if "lora_A" in key)
        next(iter(tensors.values())).fill_(7)
        after = next(value for key, value in self.state.items() if "lora_A" in key)
        self.assertTrue(torch.equal(before, after))

    def test_reject_missing_tensor(self):
        del self.state[next(key for key in self.state if "lora_A" in key)]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.extract()

    def test_reject_nan(self):
        self.state[next(key for key in self.state if "lora_B" in key)][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid adapter"):
            self.extract()

    def test_reject_wrong_shape(self):
        self.state[next(key for key in self.state if "lora_B" in key)] = torch.ones(4, 8)
        with self.assertRaisesRegex(ValueError, "Invalid adapter"):
            self.extract()

    def test_reject_multirank_checkpoint(self):
        self.fsdp["world_size"] = 2
        with self.assertRaisesRegex(ValueError, "world-size-one"):
            self.extract()

    def test_reject_nondefault_adapter(self):
        key = next(key for key in self.state if "lora_A" in key)
        self.state[key.replace(".default.", ".other.")] = self.state.pop(key)
        with self.assertRaisesRegex(ValueError, "Unexpected adapter key"):
            self.extract()

    def test_reject_dora(self):
        self.adapter["use_dora"] = True
        with self.assertRaisesRegex(ValueError, "use_dora"):
            self.extract()


if __name__ == "__main__":
    unittest.main(verbosity=2)
