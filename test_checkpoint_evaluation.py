"""CPU checks for evaluation alignment and summary interpretation."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SPEC = importlib.util.spec_from_file_location("evaluation", Path(__file__).parent / "scripts/evaluate_checkpoint.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_small_target_drift_is_recorded_not_called_exact(self):
        import torch
        cached = torch.tensor([0.25, 0.0], dtype=torch.bfloat16)
        saved = cached.float()
        saved[0] += 2.9802322387695312e-8
        saved[1] = 1e-10
        result = evaluation.compare_base_tensor(saved, cached, True, "bfloat16")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["raw_different_elements"], 2)
        self.assertEqual(result["rollout_different_elements"], 1)

    def test_larger_or_non_target_drift_is_rejected(self):
        import torch
        cached = torch.tensor([0.0])
        large = evaluation.compare_base_tensor(torch.tensor([2e-7]), cached, True, "bfloat16")
        other = evaluation.compare_base_tensor(torch.tensor([1e-10]), cached, False, "bfloat16")
        self.assertFalse(large["accepted"])
        self.assertFalse(other["accepted"])

    def test_no_relative_tolerance_for_large_weights(self):
        import torch
        result = evaluation.compare_base_tensor(torch.tensor([1000.0001]), torch.tensor([1000.0]), True, "bfloat16")
        self.assertFalse(result["accepted"])

    def test_fp32_epsilon_extension_requires_exact_rollout_tensor(self):
        import torch
        cached = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        saved = cached.float()
        saved[0] += torch.finfo(torch.float32).eps
        result = evaluation.compare_base_tensor(saved, cached, True, "bfloat16")
        self.assertTrue(result["accepted"])
        self.assertTrue(result["rollout_exact_extension_used"])
        self.assertEqual(result["rollout_different_elements"], 0)
        saved[1] = 1e-10
        result = evaluation.compare_base_tensor(saved, cached, True, "bfloat16")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["rollout_exact_extension_used"])

    def test_extension_rejects_larger_drift_even_when_bf16_matches(self):
        import torch
        cached = torch.tensor([1.0], dtype=torch.bfloat16)
        saved = cached.float() + 2 * torch.finfo(torch.float32).eps
        result = evaluation.compare_base_tensor(saved, cached, True, "bfloat16")
        self.assertEqual(result["rollout_different_elements"], 0)
        self.assertFalse(result["accepted"])

    def test_extension_preserves_target_dtype_and_rollout_guards(self):
        import torch
        cached = torch.tensor([1.0], dtype=torch.bfloat16)
        saved = cached.float() + torch.finfo(torch.float32).eps
        for value, target, dtype in [(saved, False, "bfloat16"),
                                     (saved.double(), True, "bfloat16"),
                                     (saved, True, "float32")]:
            with self.subTest(dtype=dtype, target=target, saved_dtype=value.dtype):
                result = evaluation.compare_base_tensor(value, cached, target, dtype)
                self.assertFalse(result["accepted"])

    def test_base_shape_and_finiteness_are_required(self):
        import torch
        with self.assertRaisesRegex(ValueError, "shapes"):
            evaluation.compare_base_tensor(torch.zeros(2), torch.zeros(1), True, "bfloat16")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluation.compare_base_tensor(torch.tensor([float("nan")]), torch.zeros(1), True, "bfloat16")
        with self.assertRaisesRegex(ValueError, "dtype"):
            evaluation.compare_base_tensor(torch.zeros(1), torch.zeros(1), True, "auto")

    def test_seed_alignment_survives_finished_episodes(self):
        calls = []
        request = SimpleNamespace(lora_int_id=123)
        def generate(prompts, **kwargs):
            calls.append(kwargs)
            return [object() for _ in prompts]
        engine = SimpleNamespace(generate=generate, llm_engine=SimpleNamespace(list_loras=lambda: [123]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            wrapper = evaluation.SeededEngine(engine, request, 40, Path(directory) / "trace.jsonl")
            wrapper.env_ids = [0, 1, 2]
            wrapper.generate(["a", "b", "c"], SimpleNamespace(seed=None))
            wrapper.env_ids = [2, 0]
            wrapper.generate(["c2", "a2"], SimpleNamespace(seed=None))
            self.assertEqual([p.seed for p in calls[0]["sampling_params"]], [evaluation.request_seed(40, i, 0) for i in (0, 1, 2)])
            self.assertEqual([p.seed for p in calls[1]["sampling_params"]], [evaluation.request_seed(40, i, 1) for i in (2, 0)])
            self.assertIs(calls[0]["lora_request"], request)

    def test_seed_domains_do_not_repeat_the_legacy_turn_overlap(self):
        old = evaluation.request_seed(40, 2, 1, "legacy-linear-v1")
        self.assertEqual(old, evaluation.request_seed(41, 2, 0, "legacy-linear-v1"))
        first = {evaluation.request_seed(40, i, t) for i in range(250) for t in range(32)}
        second = {evaluation.request_seed(41, i, t) for i in range(250) for t in range(32)}
        self.assertEqual(len(first), 8000)
        self.assertEqual(len(second), 8000)
        self.assertFalse(first & second)
        self.assertTrue(all(0 <= seed < 2**63 for seed in first | second))

    def test_invalid_seed_domain_is_rejected(self):
        for args in [(-1, 0, 0), (1, -1, 0), (1, 0, -1), (1, 0, 0, "unknown")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluation.request_seed(*args)

    def test_legacy_seed_protocol_remains_explicitly_available(self):
        self.assertEqual(evaluation.request_seed(40, 2, 1, "legacy-linear-v1"), 2041)

    def test_missing_adapter_registration_is_rejected(self):
        engine = SimpleNamespace(generate=lambda *a, **k: [object()], llm_engine=SimpleNamespace(list_loras=lambda: []))
        wrapper = evaluation.SeededEngine(engine, SimpleNamespace(lora_int_id=123), 1, Path("unused"))
        wrapper.env_ids = [0]
        with self.assertRaisesRegex(ValueError, "registered adapters"):
            wrapper.generate(["prompt"], SimpleNamespace(seed=None))

    def test_wilson_keeps_uncertainty_for_zero_success(self):
        low, high = evaluation.wilson(0, 50)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0.07)
        self.assertLess(high, 0.08)

    def test_configuration_preserves_training_protocol(self):
        recorded = {"custom_envs": {"BlockingWarmupRight": {"env_instruction": "exact prompt", "env_config": {"layout": "warmup_right", "max_rounds": 32}}}, "agent_proxy": {"enable_think": False, "max_context_window": 2}, "es_manager": {"val": {}}, "actor_rollout_ref": {"model": {}, "rollout": {"val_kwargs": {}}}}
        config = evaluation.evaluation_config(recorded, ["warmup_right", "blocking_short"], 50, "model", 8)
        self.assertEqual(config["custom_envs"]["PairedEval1"]["env_instruction"], "exact prompt")
        self.assertEqual(config["custom_envs"]["PairedEval1"]["env_config"]["max_rounds"], 32)
        self.assertEqual(config["agent_proxy"], recorded["agent_proxy"])
        self.assertNotIn("PairedEval1", recorded["custom_envs"])
        self.assertEqual(config["es_manager"]["val"]["group_size"], 50)

    def test_summary_distinguishes_unresolved_and_checks_prompts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for model in ("base", "adapter"):
                (root / model).mkdir()
                episodes = []
                for i, outcome in enumerate(["A", "unresolved"]):
                    episodes.append(dict(layout="warmup_right", result=outcome, resolved=outcome != "unresolved", episode_index=i, known_action_validity_count=1, invalid_parsed_action_count=0, observed_return=1 if outcome == "A" else 0, block_count=0, format_error_count=0, executed_actions=["move right"]))
                evaluation.write_json(root / model / "summary.json", {"episodes": episodes})
                (root / model / "generation-trace.jsonl").write_text("\n".join(json.dumps(dict(env_id=i, turn=0, seed=40 + i * 1000, prompt_sha256="same")) for i in range(2)))
            evaluation.comparison(root, ["warmup_right"], 2)
            result = json.loads((root / "comparison.json").read_text())
            self.assertEqual(result["rows"][0]["unresolved"], 1)
            self.assertEqual(result["rows"][0]["success_rate"], 0.5)
            target = root / "adapter/generation-trace.jsonl"
            target.write_text(target.read_text().replace('"same"', '"different"'))
            with self.assertRaisesRegex(ValueError, "Initial prompts"):
                evaluation.comparison(root, ["warmup_right"], 2)


if __name__ == "__main__":
    unittest.main()
