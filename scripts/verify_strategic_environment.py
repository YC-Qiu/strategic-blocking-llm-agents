"""Verify the custom environment's actual RAGEN registration and rule contract."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(ragen_dir=None):
    from demo import run, direct, detour, SLIDE_MAP, COSTLY_MAP
    from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv
    registration = {"checked": False}
    if ragen_dir is not None:
        sys.path.insert(0, str(ragen_dir))
        import ragen.env as registry
        assert registry.REGISTERED_ENVS["strategic_blocking"] is StrategicBlockingEnv
        assert registry.REGISTERED_ENV_CONFIGS["strategic_blocking"] is StrategicBlockingConfig
        registration = {"checked": True, "module": registry.__file__, "key": "strategic_blocking"}
    cases = []
    for layout, grid in (("slides", SLIDE_MAP), ("costly", COSTLY_MAP)):
        for policy_name, policy in (("direct", direct), ("scripted_detour", detour((5, 7), (5, 6)))):
            scripted = run(grid, policy, max_rounds=32)
            env = StrategicBlockingEnv(StrategicBlockingConfig(layout=layout))
            initial = env.reset(seed=123, mode="val")
            for turn in scripted["turns"]:
                action = turn["scripted_action"]
                if action is None:
                    continue
                assert env.race.observe() == turn["observation"]
                observation, reward, done, info = env.step(action)
                assert observation == env.render()
            assert env.race.result == scripted["summary"]["result"]
            assert env.race.reward == scripted["summary"]["reward"]
            cases.append({"layout": layout, "policy": policy_name,
                          "result": env.race.result, "reward": env.race.reward,
                          "adapter_matches_native_replay": True})
            assert env.reset(seed=123, mode="train") == initial
    env = StrategicBlockingEnv(StrategicBlockingConfig(max_rounds=1))
    env.reset()
    _, reward, done, info = env.step("not a legal action")
    assert info["executed_action"] == "wait" and not info["action_is_valid"]
    assert done and reward == 0 and info["result"] == "timeout"
    return {"registration": registration, "scripted_cases": cases,
            "invalid_parsed_action_consumes_turn": True, "deterministic_reset": True,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in ("maze.py", "demo.py", "curriculum.py", "ragen_maze.py", "strategic_blocking.yaml")},
            "scope": "Environment integration and scripted controls; these checks are not learned-policy evaluation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ragen-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.ragen_dir.expanduser().resolve() if args.ragen_dir else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Environment contract passed; actual RAGEN registry checked: {result['registration']['checked']}")
