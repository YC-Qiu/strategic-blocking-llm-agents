# RAGEN-2 integration

Run `python -m unittest -v` in this project first. The adapter uses the existing race and maps. The two fixtures and scripted replays verify the environment contract locally. Real model rollouts and training require an installed RAGEN GPU environment.

## Install and register in a separate RAGEN checkout

On a Linux CUDA machine, run the following in Bash. Set PROJECT to the absolute path of this project on that machine.

```bash
export PROJECT=/absolute/path/to/strategic-blocking-llm-agents
git clone https://github.com/mll-lab-nu/RAGEN.git
cd RAGEN
git rev-parse HEAD             # record this revision with the run logs
conda create -n ragen python=3.12 -y
conda activate ragen
bash scripts/setup_ragen.sh
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
cp "$PROJECT/strategic_blocking.yaml" config/strategic_blocking.yaml
```

Append the following three lines once to the RAGEN checkout's `ragen/env/__init__.py`. The registry is loaded inside each worker. Keep the project on PYTHONPATH in every launch shell and worker.

```python
from ragen_maze import StrategicBlockingEnv, StrategicBlockingConfig
REGISTERED_ENVS["strategic_blocking"] = StrategicBlockingEnv
REGISTERED_ENV_CONFIGS["strategic_blocking"] = StrategicBlockingConfig
```

## Verification and first rollout

From the RAGEN checkout, verify both actual registry entries before loading a model:

```bash
python - <<'PY'
from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
key = "strategic_blocking"
env = REGISTERED_ENVS[key](REGISTERED_ENV_CONFIGS[key]())
print(env.reset(seed=42, mode="val"))
print(env.step("wait"))
PY

python -m ragen.llm_agent.agent_proxy --config-name strategic_blocking \
    model_path=Qwen/Qwen2.5-0.5B-Instruct
```

The second command is the **before-RL evaluation**. Inspect `results/strategic_blocking` for the two trajectories and check the raw model output and executed actions. For a first training update on a suitably sized GPU allocation:

```bash
python train.py --config-name strategic_blocking \
    model_path=Qwen/Qwen2.5-0.5B-Instruct trainer.total_training_steps=1
```

The hardware, batches and optimizer are inherited from upstream base. Set them for your allocation before launch. Confirm that an update actually ran and save the training log before reporting loss/reward curves. Re-evaluate a saved checkpoint using the same evaluation settings and a different output filename.

## Design choices for this pilot

- One model response selects one A action. Reset advances B first. Step applies A and immediately advances B unless the race ended. Success means an A victory. The 32-round limit covers the 22-turn slide reproduction.
- A parsed but illegal action consumes a turn as a wait and is marked invalid. RAGEN can intercept malformed responses before calling step; monitor manager-invalid turns separately from environment timeouts. The format penalty is zero so the environment supplies the only win/loss reward.
- Text observations show the map, both positions and goals, legal actions, round and block status. The baseline retains two turns of history. Report the same context settings for before- and after-RL comparisons.
- Both tags reference fixed demo layouts. Validation on these fixtures checks integration only. Add separately generated layouts before claiming held-out generalization.
