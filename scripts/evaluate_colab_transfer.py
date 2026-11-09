"""Evaluate frozen checkpoints with a separate Transformers/PEFT protocol.

This backend is not a reproduction of the RAGEN/vLLM sampled evaluation.
No optimizer, gradient computation or adapter merging is used.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frozen_eval import load_suite, SUITE_SHA256
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv
from evaluate_checkpoint import wilson


def read_protocol():
    raw = (ROOT / "evaluation/colab-protocol-v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != "a6590e066f664812752770202a947f80a4f7652c43ec2d1903fd3198bb7a19f1":
        raise ValueError("Frozen Colab protocol was modified; create a new version.")
    return json.loads(raw)


def request_seed(seed, layout, episode, turn):
    key = f"hf-colab-v1/{seed}/{layout}/{episode}/{turn}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def messages_for(observation, history, protocol):
    messages = [{"role": "system", "content": protocol["instruction"]}]
    for item in history[-protocol["history_turns"]:]:
        messages.extend([{"role": "user", "content": item["observation"]},
                         {"role": "assistant", "content": item["response"]}])
    messages.append({"role": "user", "content": observation})
    return messages


def parse_action(response):
    answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if len(answers) != 1 or not answers[0].strip():
        return "__format_error__", True
    return answers[0].strip().lower(), False


def verify_adapters(bundle, suite):
    for policy, spec in suite["policies"].items():
        for name, expected in spec["files"].items():
            path = bundle / "adapters" / policy / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Unexpected frozen adapter file: {path}")


def run_episode(layout, policy, index, generate, protocol, seed):
    env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout, max_rounds=32))
    observation = env.reset()
    history = []
    while env.race.result is None:
        turn = len(history)
        sampled_seed = request_seed(seed, layout, index, turn)
        messages = messages_for(observation, history, protocol)
        response = generate(messages, sampled_seed)
        action, format_error = parse_action(response)
        following, reward, _, info = env.step(action)
        history.append({"observation": observation, "response": response,
                        "seed": sampled_seed, "format_error": format_error,
                        "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                        "executed_action": info["executed_action"],
                        "action_is_valid": info["action_is_valid"], "reward": reward})
        observation = following
    return {"layout": layout, "policy": policy, "episode": index,
            "result": env.race.result, "history": history, "final_observation": observation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One episode per policy on an old training fixture.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    suite, protocol = load_suite(), read_protocol()
    verify_adapters(args.bundle, suite)
    if args.verify_only:
        print("Frozen suite and both adapter file hashes verified; no model loaded.")
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before starting this evaluation.")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="colab-smoke-" if args.smoke else "colab-transfer-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(suite["model_id"], revision=suite["base_revision"])
    base = AutoModelForCausalLM.from_pretrained(suite["model_id"], revision=suite["base_revision"],
                                               torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, args.bundle / "adapters/before", adapter_name="before", is_trainable=False)
    model.load_adapter(args.bundle / "adapters/after", adapter_name="after", is_trainable=False)
    model.eval()
    layouts = ["warmup_right"] if args.smoke else [item["name"] for item in suite["maps"]]
    count = 1 if args.smoke else suite["episodes_per_map"]
    manifest = {"scope": "Separate Transformers/PEFT evaluation; do not pool with RAGEN results.",
                "smoke_only": args.smoke, "suite_sha256": SUITE_SHA256,
                "protocol": protocol, "layouts": layouts, "episodes_per_map": count,
                "experiment_seed": suite["experiment_seed"], "base_revision": suite["base_revision"],
                "adapters": suite["policies"], "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(), "dtype": "float16",
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ["maze.py", "ragen_maze.py", "frozen_eval.py", "scripts/evaluate_colab_transfer.py", "evaluation/colab-protocol-v1.json"]}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "dependencies.txt").write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    summaries = []
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for policy in ("base", "before", "after"):
            if policy != "base":
                model.set_adapter(policy)
            model.eval()
            context = model.disable_adapter() if policy == "base" else nullcontext()
            with context, torch.inference_mode():
                def generate(messages, seed):
                    torch.manual_seed(seed)
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                           return_tensors="pt", return_dict=True).to("cuda")
                    if inputs["input_ids"].shape[1] + protocol["max_new_tokens"] > 2048:
                        raise ValueError("Prompt exceeds the frozen context budget; no silent truncation.")
                    ids = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0,
                                         top_k=0, max_new_tokens=protocol["max_new_tokens"],
                                         pad_token_id=tokenizer.eos_token_id)
                    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for layout in layouts:
                    wins, invalid, actions = 0, 0, 0
                    outcomes = {"A": 0, "B": 0, "timeout": 0}
                    for index in range(count):
                        episode = run_episode(layout, policy, index, generate, protocol, suite["experiment_seed"])
                        stream.write(json.dumps(episode) + "\n")
                        stream.flush()
                        wins += episode["result"] == "A"
                        outcomes[episode["result"]] += 1
                        invalid += sum(not h["action_is_valid"] for h in episode["history"])
                        actions += len(episode["history"])
                    summaries.append(dict(layout=layout, policy=policy, wins=wins, episodes=count,
                                          success_rate=wins/count, wilson95=wilson(wins, count),
                                          invalid_actions=invalid, actions=actions, outcomes=outcomes))
                    print(f"{policy}: {layout}: {wins}/{count}", flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = ["# Colab transfer evaluation", "", "Separate backend and protocol; do not pool with RAGEN results.", "",
             "| Layout | Policy | Wins | Success | 95% Wilson interval |", "|---|---|---:|---:|---:|"]
    for row in summaries:
        lo, hi = row["wilson95"]
        lines.append(f"| {row['layout']} | {row['policy']} | {row['wins']}/{count} | {row['success_rate']:.1%} | {lo:.1%}-{hi:.1%} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "COMPLETE").write_text("All declared policies and episodes completed.\n")
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
