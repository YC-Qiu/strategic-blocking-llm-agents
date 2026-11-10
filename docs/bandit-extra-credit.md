# Additional environment: native RAGEN Bandit

## Status and purpose

Run `train-bandit-0WF04q1N` completed five updates, each with nonzero
advantages, checkpoint export, loss/reward curves, and matched evaluation of
200 base and 200 adapter episodes plus four controls. A CPU record audit
recomputed the summaries and matched prompts, seeds, and recorded payouts.
High-arm choice was 50.0% for base and 46.5% for adapter: no improvement is shown.
See [audited results](bandit-results.md) and [audit evidence](bandit-results-audit.json).
The experiment supplies additional-environment/baseline evidence independently
of whether RL improved this pilot's policy.

The commands below reproduce a new run; they are not required to finish the
already completed pilot. Preserve the existing results rather than rerunning
only to obtain a positive outcome.

## Frozen pilot design

- Start from Qwen2.5-0.5B-Instruct with a fresh rank-8, alpha-16 LoRA. Never load
  a maze adapter, optimizer or checkpoint. Automatic resume is disabled.
- Use the installed native Bandit environment, reward distribution and name splits.
- Run five GRPO updates, two groups of eight samples per update: 80 training
  episodes. Validation uses 16 training-name instances, greedy decoding, before
  training and after each update. Validation is not the final test set.
- Evaluate base and final adapter on 200 identical test-name instances each,
  with sampled decoding (temperature 1, top-p 1, no top-k cutoff), one response
  per episode, 64 generated tokens and a 2048-token context limit.
- Use the actual installed RAGEN context manager, including its assistant
  prefill and response reconstruction. Do not reuse a hand-written Colab prompt.
- Match initial prompt hashes, environment seeds, native instance seeds,
  generation seeds and control outcomes across the two isolated model processes.
- More than 5% invalid model decisions stops the pipeline. Invalid decisions
  have no executed action and reward zero; they are not silently mapped to an arm.
- One fixed training seed is a pilot, not a robustness study. Do not rerun or
  select checkpoints merely to obtain a positive result.

## What the controls measure

The conservative arm always gives 0.1. The high-expectation arm gives 1 with
probability 0.25 and otherwise 0, so its expected reward is 0.25. Uniform random
selection has expected reward 0.175. A correct high-arm choice can receive zero.

| Policy | Information used | Purpose |
|---|---|---|
| Base LLM | Native observation and prompt | Before-RL baseline |
| Trained LoRA | Same observation and prompt | After-RL policy |
| Uniform random | Available actions only | Chance control |
| First position | First displayed arm only | Position-bias control |
| Always low | Privileged low-arm identity | Conservative reference |
| High oracle | Privileged high-arm identity | Optimal expected-reward reference |

The latter two are simulator-aware controls, not learned policies. The summary
reports high-arm selection, observed reward, expected reward from the known
distribution, expected regret, invalid count, and a Wilson interval for high-arm
selection. Do not interpret a few lucky payouts as improved decision quality.

## Native implementation caveats

Bandit adjusts reset seeds using a class-level instance counter. All comparisons
therefore use fresh processes, the same environment construction order, and
explicitly check both seed values and initial observations. Each control replays
the exact same native instance/seed and reward draw. Training preserves upstream
behavior: members of one GRPO group can receive different name observations.
This is a native-framework integration pilot, not a controlled same-prompt GRPO
ablation or an exploration study across repeated pulls.

The installed manager fills an absent `info.raw_reward` with zero. Native Bandit
does not supply this field, so `train/charts/avg_episode_return` can misleadingly
show zero. This workflow plots `critic/score/mean`, which is derived from actual
history rewards under identity normalization and a zero format penalty. Raw
evaluation rewards are checked by replay. No reward rule is patched.

## Commands on the existing Ubuntu installation

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -m unittest discover -q
bash scripts/run_bandit_bonus.sh --check-only
bash scripts/run_bandit_bonus.sh
```

Check-only verifies the actual environment registry, native reward behavior,
name splits, deterministic same-instance replay and composed configuration. It
does not train or load an LLM. The existing guarded compatibility patch script
is checked/applied in both modes. No dependency upgrade is performed.

The full command runs base evaluation, training, checkpoint audit/export, adapter
evaluation and report generation sequentially. A failed stage leaves FAILED.json
and logs under a new `results/bandit/train-bandit-*` directory. It does not resume
an incomplete directory or overwrite previous experiments.

Important outputs:

- `run.log`, `composed.yaml`, dependencies, source hashes and RAGEN revision.
- `base/episodes.jsonl`, `adapter/episodes.jsonl` and generation traces.
- `bandit-metrics.json`, `bandit-learning-curves.png` and `.pdf`.
- `comparison.json`, `comparison.md`, and `adapter-update-check.json`.
- `environment-contribution.json` verifying actual custom-environment registration.

The reward/loss plots require their actual logged keys; missing metrics cause an
explicit failure rather than an empty or invented curve. A nonzero LoRA B norm
establishes parameter-change evidence for fresh standard LoRA, not improved policy
quality. The report must distinguish five recorded updates from early stopping.

Keep checkpoints, raw logs and adapters under ignored results/. Once the run has
been audited, copy compact result tables/figures into docs/ for the final report.
No commit, push, upstream submission or external message is automated.
