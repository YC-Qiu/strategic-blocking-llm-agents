"""Native RAGEN Bandit preflight, matched policy evaluation and result reporting."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_checkpoint import (SeededEngine, file_hash, load_recorded_config,
                                 prepare_models, wilson, write_json)
from summarize_training import parse_metrics

EVAL_SEED = 20261117
EVAL_COUNT = 200


def expected_reward(choice, low=0.1, high_probability=0.25, high_win=1.0, high_loss=0.0):
    high = high_probability * high_win + (1 - high_probability) * high_loss
    return {"low": low, "high": high, "invalid": 0.0}[choice]


def metrics(rows):
    if not rows:
        raise ValueError("No Bandit episodes to summarize.")
    n = len(rows)
    high = sum(r["choice"] == "high" for r in rows)
    return {"episodes": n, "high_arm_choices": high, "high_arm_rate": high / n,
            "high_arm_wilson95": wilson(high, n),
            "invalid": sum(r["choice"] == "invalid" for r in rows),
            "mean_observed_reward": sum(r["reward"] for r in rows) / n,
            "mean_expected_reward": sum(r["expected_reward"] for r in rows) / n,
            "mean_expected_regret": sum(0.25-r["expected_reward"] for r in rows) / n}


def make_evaluation_config(recorded, snapshot, count=EVAL_COUNT):
    cfg = deepcopy(recorded)
    cfg["model_path"] = str(snapshot)
    cfg["actor_rollout_ref"]["model"]["path"] = str(snapshot)
    cfg["actor_rollout_ref"]["model"].pop("lora_adapter_path", None)
    cfg["seed"]["val"] = EVAL_SEED
    cfg["es_manager"]["val"] = dict(env_groups=count, group_size=1,
        env_configs=dict(tags=["BonusBanditTest"], n_groups=[count]))
    rollout = cfg["actor_rollout_ref"]["rollout"]
    rollout.update(max_num_seqs=4, seed=EVAL_SEED, enable_chunked_prefill=False)
    rollout["val_kwargs"].update(do_sample=True, temperature=1.0, top_p=1.0, top_k=-1)
    return cfg


def install_import_path(ragen):
    sys.path.insert(0, str(ragen))


def preflight(args):
    install_import_path(args.ragen_dir)
    import matplotlib  # Fail before training if the curve dependency is missing.
    from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
    from ragen.env.bandit.config import ARM_NAMES
    from ragen_maze import StrategicBlockingEnv, StrategicBlockingConfig
    assert REGISTERED_ENVS["strategic_blocking"] is StrategicBlockingEnv
    assert REGISTERED_ENV_CONFIGS["strategic_blocking"] is StrategicBlockingConfig
    train_names = {name for pair in ARM_NAMES["train"] for name in pair}
    test_names = {name for pair in ARM_NAMES["test"] for name in pair}
    assert not train_names & test_names, "Native train/test arm names overlap."
    env = REGISTERED_ENVS["bandit"](REGISTERED_ENV_CONFIGS["bandit"](split="test"))
    cfg = env.config
    assert (cfg.lo_arm_score, cfg.hi_arm_loscore, cfg.hi_arm_hiscore, cfg.hi_arm_hiscore_prob) == (0.1, 0.0, 1.0, 0.25)
    examples = []
    for seed in range(32):
        observation = env.reset(seed=seed)
        low, high = env.lo_arm_name, env.hi_arm_name
        first = env.step(env.NAME_TO_ARM_IDX[high])
        assert env.reset(seed=seed) == observation
        assert env.step(env.NAME_TO_ARM_IDX[high]) == first
        assert first[1] in (0.0, 1.0) and first[2] and first[3]["success"]
        env.reset(seed=seed)
        assert env.step(env.NAME_TO_ARM_IDX[low])[1] == 0.1
        if seed < 3:
            examples.append({"seed": seed, "observation": observation, "low": low, "high": high})
    report = {"native_environment_checks": "passed", "same_instance_seed_replay": "passed",
              "train_test_names_disjoint": True, "train_name_pairs": len(ARM_NAMES["train"]),
              "test_name_pairs": len(ARM_NAMES["test"]),
              "analytical_expected_rewards": {"low": 0.1, "high_oracle": 0.25, "uniform_random": 0.175},
              "examples": examples,
              "native_bandit_sources": {name: file_hash(args.ragen_dir / "ragen/env/bandit" / name)
                                        for name in ("env.py", "config.py")},
              "note": "Native instance-dependent seeds are preserved; GRPO groups may contain different arm-name observations."}
    write_json(args.run / "bandit-preflight.json", report)
    env.close()
    print("PASS: native Bandit rewards, seed replay, disjoint names and Strategic Blocking registration.")


def prepare(args):
    from huggingface_hub import snapshot_download
    import yaml
    cfg = load_recorded_config(args.run / "composed.yaml")
    if cfg["actor_rollout_ref"]["model"].get("lora_adapter_path"):
        raise ValueError("Bandit must start from the original base, not a maze adapter.")
    snapshot = Path(snapshot_download(cfg["model_path"], local_files_only=True))
    eval_cfg = make_evaluation_config(cfg, snapshot)
    (args.run / "evaluation-config.yaml").write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")
    write_json(args.run / "evaluation-plan.json", {
        "protocol": "native-ragen-bandit-v1", "episodes_per_model": EVAL_COUNT, "seed": EVAL_SEED,
        "base_snapshot": str(snapshot), "base_revision": snapshot.name,
        "format_gate_max_invalid_rate": 0.05,
        "policies": ["base", "adapter", "uniform_random", "first_position", "always_low", "high_oracle"],
        "scope": "One training seed; unseen arm names within the native Bandit environment.",
        "evaluation_config_sha256": file_hash(args.run / "evaluation-config.yaml")})


def export(args):
    recorded = load_recorded_config(args.run / "composed.yaml")
    steps = recorded["trainer"]["total_training_steps"]
    actor = args.run / f"checkpoints/global_step_{steps}/actor"
    plan = json.loads((args.run / "evaluation-plan.json").read_text())
    snapshot, adapter = prepare_models(actor, args.run, recorded["actor_rollout_ref"]["rollout"]["dtype"])
    if str(snapshot) != plan["base_snapshot"]:
        raise ValueError("Export selected a different cached base than the baseline evaluation.")
    from safetensors.torch import load_file
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    b_norm = sum(float(value.float().square().sum()) for key, value in weights.items() if "lora_B" in key)
    if not b_norm > 0:
        raise ValueError("Exported LoRA B tensors remain zero; no parameter-update evidence.")
    write_json(args.run / "adapter-update-check.json", {
        "lora_b_squared_norm": b_norm, "nonzero": True,
        "interpretation": "Fresh standard LoRA initializes B to zero; this check establishes changed B tensors, not policy improvement."})


def worker(args):
    install_import_path(args.ragen_dir)
    import yaml
    config_dict = yaml.safe_load((args.run / "evaluation-config.yaml").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["system"]["CUDA_VISIBLE_DEVICES"])
    import ragen.llm_agent.agent_proxy as proxy_module
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest
    from verl import DataProto
    from probe_reward_signal import json_default
    plan = json.loads((args.run / "evaluation-plan.json").read_text())
    if file_hash(args.run / "evaluation-config.yaml") != plan["evaluation_config_sha256"]:
        raise ValueError("Frozen evaluation config was modified.")
    config = OmegaConf.create(config_dict)
    destination = args.run / args.policy
    destination.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    request = None if args.policy == "base" else LoRARequest("bandit-final", 123, str(args.run / "verified_adapter"))
    original = proxy_module.LLM

    def make_llm(*positional, **kwargs):
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=8, seed=EVAL_SEED, max_num_seqs=4)
        return original(*positional, **kwargs)

    class EvaluationWorker(proxy_module.VllmWrapperWg):
        def generate_sequences(self, inputs):
            self.llm.env_ids = [int(i) for i in inputs.non_tensor_batch["env_ids"]]
            return super().generate_sequences(inputs)

    proxy_module.LLM = make_llm
    try:
        actor = EvaluationWorker(config, tokenizer)
    finally:
        proxy_module.LLM = original
    actor.llm = SeededEngine(actor.llm, request, EVAL_SEED, destination / "generation-trace.jsonl")
    proxy = proxy_module.LLMAgentProxy(config, actor, tokenizer)
    rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False, "do_sample": True, "validate": True}), val=True)
    histories = list(rollouts.non_tensor_batch["history"])
    entries = proxy.val_es_manager.envs
    if len(histories) != EVAL_COUNT or len(entries) != EVAL_COUNT:
        raise ValueError("Unexpected evaluation episode count.")
    records, controls = [], []
    for index, (history, entry) in enumerate(zip(histories, entries)):
        env = entry["env"]
        reset_seed = entry["status"].seed
        observation = env.reset(seed=reset_seed)
        if observation != history[0]["state"]:
            raise ValueError("Native environment seed replay differs from the model's observation.")
        actions = [a for turn in history for a in turn.get("actions", [])]
        reward = sum(float(turn.get("reward", 0.0)) for turn in history)
        if len(actions) > 1:
            raise ValueError("Expected at most one executed Bandit action.")
        choice = "invalid" if not actions else ("high" if env.ARM_IDX_TO_NAME[int(actions[0])] == env.hi_arm_name else "low")
        if actions and abs(float(env.step(int(actions[0]))[1])-reward) > 1e-8:
            raise ValueError("Reward replay differs from recorded model trajectory.")
        record = {"episode": index, "policy": args.policy, "choice": choice, "reward": reward,
                  "expected_reward": expected_reward(choice), "reset_seed": reset_seed,
                  "internal_seed": env.internal_seed, "low_name": env.lo_arm_name, "high_name": env.hi_arm_name,
                  "observation_sha256": hashlib.sha256(observation.encode()).hexdigest(), "history": history}
        records.append(record)
        for control in ("uniform_random", "first_position", "always_low", "high_oracle"):
            env.reset(seed=reset_seed)
            if control == "uniform_random":
                key = f"bandit-controls-v1/{EVAL_SEED}/{index}"
                control_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                action = random.Random(control_seed).choice(env.get_all_actions())
            elif control == "first_position":
                action = min(env.get_all_actions())
            else:
                action = env.NAME_TO_ARM_IDX[env.lo_arm_name if control == "always_low" else env.hi_arm_name]
            selected = "high" if env.ARM_IDX_TO_NAME[action] == env.hi_arm_name else "low"
            controls.append({"episode": index, "policy": control, "choice": selected,
                             "reward": float(env.step(action)[1]), "expected_reward": expected_reward(selected)})
    with (destination / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, default=json_default) + "\n")
    write_json(destination / "controls.json", controls)
    result = metrics(records)
    result["format_gate_passed"] = result["invalid"] / EVAL_COUNT <= plan["format_gate_max_invalid_rate"]
    write_json(destination / "summary.json", result)
    if not result["format_gate_passed"]:
        raise ValueError(f"{args.policy} invalid-output rate exceeds 5%; stop and inspect raw responses.")
    print(f"{args.policy}: high-arm choices={result['high_arm_rate']:.1%}, mean reward={result['mean_observed_reward']:.4f}")


def report(args):
    all_rows = {}
    for policy in ("base", "adapter"):
        all_rows[policy] = [json.loads(line) for line in (args.run / policy / "episodes.jsonl").read_text().splitlines()]
    keys = ("episode", "reset_seed", "internal_seed", "low_name", "high_name", "observation_sha256")
    base_instances = [tuple(row[k] for k in keys) for row in all_rows["base"]]
    adapter_instances = [tuple(row[k] for k in keys) for row in all_rows["adapter"]]
    if base_instances != adapter_instances:
        raise ValueError("Base and adapter environments are not matched.")
    for policy, rows in all_rows.items():
        if len(rows) != EVAL_COUNT or [r["episode"] for r in rows] != list(range(EVAL_COUNT)):
            raise ValueError("Incomplete or duplicate evaluation episodes.")
        if metrics(rows)["invalid"] / EVAL_COUNT > 0.05:
            raise ValueError(f"{policy} evaluation failed the invalid-output gate.")
    traces = {}
    for policy in ("base", "adapter"):
        traces[policy] = [json.loads(line) for line in (args.run / policy / "generation-trace.jsonl").read_text().splitlines()]
    trace_keys = ("env_id", "turn", "seed", "seed_scheme", "prompt_sha256")
    for rows in traces.values():
        if len(rows) != EVAL_COUNT or {r["env_id"] for r in rows} != set(range(EVAL_COUNT)):
            raise ValueError("Incomplete or duplicate generation traces.")
    if [tuple(r[k] for k in trace_keys) for r in traces["base"]] != [tuple(r[k] for k in trace_keys) for r in traces["adapter"]]:
        raise ValueError("Generation prompts or sampling seeds do not match.")
    controls = json.loads((args.run / "base/controls.json").read_text())
    if controls != json.loads((args.run / "adapter/controls.json").read_text()):
        raise ValueError("Matched control outcomes differ between model runs.")
    for name in ("uniform_random", "first_position", "always_low", "high_oracle"):
        all_rows[name] = [row for row in controls if row["policy"] == name]
        if [r["episode"] for r in all_rows[name]] != list(range(EVAL_COUNT)):
            raise ValueError("Incomplete matched control episodes.")
    results = {name: metrics(rows) for name, rows in all_rows.items()}
    steps = parse_metrics((args.run / "run.log").read_text(errors="replace"))
    training = [r for r in steps if "actor/pg_loss" in r]
    expected_steps = load_recorded_config(args.run / "composed.yaml")["trainer"]["total_training_steps"]
    if [int(r["step"]) for r in training] != list(range(1, expected_steps + 1)):
        raise ValueError("Requested training updates were not all recorded.")
    nonzero = sum(r.get("critic/advantages/min", 0) != 0 or r.get("critic/advantages/max", 0) != 0 for r in training)
    missing_advantages = [r["step"] for r in training if not {"critic/advantages/min", "critic/advantages/max"}.issubset(r)]
    if missing_advantages:
        raise ValueError(f"Missing advantage metrics at steps {missing_advantages}; missing is not zero.")
    early_stop = any(k.startswith("early_stopped/") and v != 0 for r in steps for k, v in r.items())
    write_json(args.run / "comparison.json", {"results": results, "updates": len(training),
        "updates_with_nonzero_advantages": nonzero, "matched_prompts_and_environment_seeds": True,
        "early_stop_observed": early_stop,
        "scope": "One training seed, native name-based Bandit; no improvement claim is inferred automatically."})
    write_json(args.run / "bandit-metrics.json", steps)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    # Native Bandit omits info.raw_reward; this installed manager logs a zero
    # charts/avg_episode_return for that environment. Score metrics use the
    # actual history reward, with identity normalization and no reward penalty.
    for axis, key, title in zip(axes, ["actor/pg_loss", "critic/score/mean", "train/BonusBanditTrain/success"],
                                 ["Policy-gradient loss", "Sampled training reward", "High-arm choice fraction"]):
        points = [(r["step"], r[key]) for r in training if key in r]
        if not points:
            raise ValueError(f"Missing Bandit curve metric: {key}; inspect bandit-metrics.json.")
        axis.plot(*zip(*points), marker="o")
        axis.set(title=title, xlabel="Training update")
        axis.grid(alpha=0.25)
    fig.savefig(args.run / "bandit-learning-curves.png", dpi=160)
    fig.savefig(args.run / "bandit-learning-curves.pdf")
    plt.close(fig)
    lines = ["# Native RAGEN Bandit pilot", "", f"Recorded updates: {len(training)}; updates with nonzero advantages: {nonzero}.", "",
             "| Policy | High-arm choice | Mean reward | Expected reward | Invalid |", "|---|---:|---:|---:|---:|"]
    for name, value in results.items():
        lines.append(f"| {name} | {value['high_arm_rate']:.1%} | {value['mean_observed_reward']:.4f} | {value['mean_expected_reward']:.4f} | {value['invalid']}/{value['episodes']} |")
    lines += ["", "High-oracle and always-low controls use privileged environment labels. Random and first-position controls do not.",
              f"Trainer early-stop flag observed: {early_stop}. Recorded updates do not alone establish normal trainer completion.",
              "Expected reward uses the known simulator distribution and separates decision quality from lucky payouts.",
              "The native reward is stochastic; success means choosing the high-expectation arm, not winning a payout.",
              "Native per-instance seeding can give different name observations within a GRPO group. This behavior was not changed.",
              "The test split contains unseen arm names. This is one training seed, not proof of broad generalization.",
              "Actions-only generation does not provide an explicit reasoning trace."]
    (args.run / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.run / "COMPLETE").write_text("Requested updates, checkpoint audit and matched evaluation completed.\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "export", "worker", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ragen-dir", type=Path, default=Path.home() / "RAGEN")
    parser.add_argument("--policy", choices=("base", "adapter"))
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.ragen_dir = args.ragen_dir.expanduser().resolve()
    if args.command == "worker" and args.policy is None:
        parser.error("worker requires --policy")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
