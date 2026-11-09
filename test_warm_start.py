"""Check that warm-start verification cannot silently accept fresh weights."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("warm_start", Path(__file__).parent / "scripts/verify_warm_start.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.digest = "a" * 64
        self.manifest = {"tensor_count": 336, "tensor_sha256": self.digest}
        self.message = "Loading pre-trained LoRA adapter to actor from: /local/adapter\n"

    def test_matching_initial_collection(self):
        result = MODULE.verify_initial_collection(self.manifest, self.message + f"PILOT_LORA_COLLECT tensors=336 sha256={self.digest}\n")
        self.assertTrue(result["first_collected_weights_match_source"])

    def test_fresh_weights_are_not_accepted(self):
        with self.assertRaises(ValueError):
            MODULE.verify_initial_collection(self.manifest, self.message + "PILOT_LORA_COLLECT tensors=336 sha256=" + "b" * 64)

    def test_later_match_does_not_hide_wrong_initial_weights(self):
        log = self.message + "PILOT_LORA_COLLECT tensors=336 sha256=" + "b" * 64
        log += f"\nPILOT_LORA_COLLECT tensors=336 sha256={self.digest}"
        with self.assertRaises(ValueError):
            MODULE.verify_initial_collection(self.manifest, log)

    def test_loading_message_is_required(self):
        with self.assertRaises(ValueError):
            MODULE.verify_initial_collection(self.manifest, f"PILOT_LORA_COLLECT tensors=336 sha256={self.digest}")


if __name__ == "__main__":
    unittest.main()
