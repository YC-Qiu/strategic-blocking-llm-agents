# Strategic Blocking RL Pilot Execution Plan

## Progress after balanced training and paired evaluation

| Task | Current evidence | Remaining scope |
| --- | --- | --- |
| A1 Local framework | Full GPU rollout/update/save runs completed | Clean-install reproducibility |
| A2 Adapter handling | Original PEFT reload checked; current automatic export and vLLM evaluation work | Independent Colab handoff |
| A3 Reward signal | Both direction groups have mixed rewards in all five balanced updates | Informative blocking curriculum |
| A4 Training curves | Five-update runs, metrics and saved checkpoints available | Longer-run stability if needed |
| A5 Improvement | Right 34% to 94%; left 2% to 64%, 50 samples each | Replication, generalization, action validity and blocking |
| A6-A9 and teammate tasks | Integration and action-only case-study groundwork exists | Handoff, extra environment, contribution packaging and final deliverables |

These navigation results are on fixed training maps. Blocking success remains
0/50 for the balanced adapter. No new Colab completion is claimed. See
[the current evidence](docs/balanced-pilot-results.md). The A0 table below records
the historical starting point; later task sections describe the intended scope,
not automatic completion of every acceptance criterion.

This plan divides the Strategic Blocking RL pilot into two coordinated workstreams. YC owns local Ubuntu training with RAGEN. Prakeerth owns Google Colab evaluation, baselines, visualizations, and case studies. Both use the same existing Qwen model and maze environment.

Qwen is the model that selects actions. RAGEN trains it. Transformers loads and runs it in Colab; PEFT loads the LoRA adapter produced by training. Prakeerth does not need to implement a Transformer or reproduce the full distributed training stack.

Repository: [YC-Qiu/strategic-blocking-llm-agents](https://github.com/YC-Qiu/strategic-blocking-llm-agents). Windows checkout: `D:\Git\strategic-blocking-llm-agents`. WSL checkout: `/mnt/d/Git/strategic-blocking-llm-agents`.

The plan targets all three required outcomes and all four bonus categories. Bonus credit depends on the instructor's assessment and actual evidence; improvement and upstream acceptance are not guaranteed. Existing commands are distinguished below from proposed files and features that still need implementation.

## Part 1 YC Local Ubuntu and RAGEN Training

### A0 Establish the starting point and grading targets

| Component | Starting evidence | Work identified at the starting point |
| --- | --- | --- |
| Environment | Maze rules, fixed BFS opponent, six scripted demonstrations, and 10 passing tests | Extend layouts and preserve rule consistency |
| RAGEN integration | Adapter and packaged launcher, compatibility patches, and preflight checks | Verify the packaged entry point with a complete GPU run |
| Before RL | Two greedy episodes using Qwen2.5 0.5B Instruct; both lost | Evaluate a larger fixed suite through the shared evaluator |
| First training run | One iteration; four losses; mean return -1; zero GRPO advantages and policy-gradient loss | Multi-iteration training and informative group reward variation |
| Weights | LoRA tensors changed in the main checkpoint | Repair empty standalone export and verify reload |
| Learning | Strategic learning is not demonstrated | Compare the original and trained models under identical evaluation conditions |
| Case studies | Scripted traces and action-only model traces | Actual model decision explanations, trajectory visualizations, and failure analysis |

The original completed run is `results/train/train-one-F5JCCw3d`. It took approximately 391 seconds for one iteration. Its weight changes are consistent with entropy-driven updates, not evidence of successful reward-driven strategic learning. A controlled ablation is needed before making a stronger causal claim.

| Rubric item | Owner and tasks | Evidence required |
| --- | --- | --- |
| Requirement 1 Run an RL framework | YC, A1–A5 | Reproducible launch, full rollout/update/save logs, valid checkpoints |
| Requirement 2 Loss and reward curves | YC produces metrics; Prakeerth plots them, A4 and B7 | Multiple training iterations, named loss terms, episode-return curves |
| Requirement 3 Case studies and thinking | Prakeerth, with YC's models, B6 | Actual model outputs, action sequences, outcomes, and interpretation |
| Bonus Interesting RL findings, 2–5 points | Both, A3 and B6–B7 | Controlled comparisons and evidence for zero-advantage behavior, timing, or explanation/action mismatch |
| Bonus More environments or baselines, 2–5 points | Both, A7 and B4/B8 | Additional baselines and a separate environment with completed runs |
| Bonus Improve the baseline, 5–20 points | YC trains; Prakeerth evaluates, A3–A5 and B5 | Improvement over the original same-size model under matched settings |
| Bonus Add a new environment to an existing repository, 5–20 points | YC packages; Prakeerth validates, A8 | RAGEN integration, configuration, tests, documentation, and contribution-ready changes |
| Report, 10 points | Both, A9 and B10 | Methods, curves, cases, results, limitations, contribution record |
| Presentation, 10 points | Both, A9 and B10 | Pilot findings plus 1–2 slides proposing the final project |

The supplied assignment lists April 28 for the report and presentation and Monday of Week 5 for the joint presentation. The year and current applicability need confirmation; use the milestone sequence below rather than treating those dates as a new schedule. Team scores normally match, with possible adjustments for individual contributions, so keep a brief contribution log.

RAGEN, VAGEN, and Search-R1 are example framework choices in the assignment. This project uses RAGEN; completing all three frameworks is not necessary.

### A1 Validate the current local workflow

Use the working Ubuntu environment rather than rebuilding it immediately. Inspect `docs/local-training.md` first. The recorded stack has dependency conflicts, so the recorded package list is evidence of one working installation, not a clean dependency lock.

Existing commands, run in Ubuntu:

~~~bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
export RAGEN_DIR="$HOME/RAGEN"
python -m unittest -v
git rev-parse HEAD
git -C "$RAGEN_DIR" rev-parse HEAD
git -C "$RAGEN_DIR" status --short
bash scripts/run_ragen.sh train-one --check-only
bash scripts/run_ragen.sh train-one
~~~

Use the actual environment name if it differs. Record the RAGEN revision together with local changes: a revision alone does not capture patched code. Preserve the patch files and the configuration generated for each run. The existing launcher modifies three external RAGEN files through guarded compatibility patches.

**Acceptance:** tests and preflight pass; a complete run performs rollout, optimization, and checkpoint saving with finite values. Label this a training smoke test even if all trajectories lose.

**Deliver:** launch instructions, package versions, RAGEN/project revisions, applied patches, configuration, and a concise run summary.

### A2 Fix adapter export before scaling training

1. Inspect the saved FSDP model checkpoint and the installed RAGEN/verl export utilities.
2. Reconstruct a complete LoRA state using a version-appropriate method. Do not rename the FSDP checkpoint to a safetensors file or assume its key names already match PEFT.
3. Save a normal PEFT adapter, including `adapter_config.json` and `adapter_model.safetensors`.
4. Load it into a fresh instance of the exact original base-model revision.
5. Check that the expected adapter tensors exist, have valid shapes and finite values, and match the checkpoint values after any required key conversion.
6. Compare fixed-prompt outputs or logits before and after export using the same backend and precision. Check numerical consistency within a documented tolerance; changing the generated action is not required.
7. Give Prakeerth this first adapter as a loading test, even if it has no demonstrated strategic improvement.

**Deliver:** adapter directory, base model/revision, tokenizer revision or files, training step, LoRA configuration, export procedure, and reload check. The current empty standalone adapter must not be used.

### A3 Make the reward signal informative

**November 8 status:** In progress. Five curriculum fixtures, a 28-episode sampled probe, baseline trajectory diagnosis, and 11 additional tests are prepared. All 28 local tests pass; GPU probe execution and informative trainer advantages remain unverified. See [the A3 runbook](docs/reward-signal-probe.md).

Keep the main experiment's reward definition: A win +1, B win -1, timeout 0, with no intermediate reward.

The current failure is that every sampled episode loses. GRPO compares outcomes within groups, so reward variation across unrelated maps is insufficient. Confirm the actual group membership and that trajectories compared within a group start from the same intended task state.

Work in this order:

1. Inspect sampled actions, malformed responses, illegal actions, repeated moves, and failure locations. Separate action-format failure from strategic failure.
2. Ask Prakeerth for small validated training maps where success is reachable with short action sequences. Include easy direct navigation, a short useful block, and a case where blocking is costly.
3. Start a curriculum on these easier tasks, then add harder blocking cases. Keep the final test maps out of training.
4. Change one variable per diagnostic experiment: task difficulty, sampling temperature, group size, or context length. Increase group size only within the measured memory budget.
5. Log the fraction of groups with nonzero reward variance, advantage statistics, and success rate.
6. If several short runs still have zero informative groups, revise the sampling/task design before committing to a long run.

A nonzero advantage is an intermediate diagnostic, not proof of useful learning. If demonstration-based supervised warm-start training is later needed, report it as a separate treatment and compare SFT-only with SFT-plus-RL. Do not attribute combined gains entirely to RL. Any reward shaping must likewise be a separately labeled ablation, with evaluation still using the original task rewards.

**Acceptance:** group statistics are understood; at least one configuration produces an informative signal, or the failure is clearly documented with bounded experiments.

### A4 Add continuous training and reliable metrics

The current `train-one` mode starts a new one-iteration run. Repeatedly invoking it does not automatically continue training.

Implement configurable training length, seed, save interval, and explicit resume behavior. Document the new interface after implementation; no multi-step or resume flags are currently assumed to exist.

Suggested progression:

- First validate one uninterrupted run of 3–5 iterations.
- Then target 20–50 iterations if memory, stability, and group reward variation justify continuing.
- Save several intermediate checkpoints. Retain at least one resumable training checkpoint with optimizer state.
- Verify a restart continues the optimizer and step counter rather than resetting the model.

These are planning targets, not guaranteed sufficient training or grading thresholds.

Export a tidy CSV or JSONL with:

| Metric group | Required fields |
| --- | --- |
| Identity | Run ID, global iteration, seed, project/RAGEN revisions, configuration |
| Rewards | Mean terminal episode return, success count, episode count, timeout count, per-layout results |
| Optimization | Policy-gradient loss, entropy term, total actor loss if available, gradient norm |
| GRPO diagnostics | Advantage minimum/mean/maximum, nonzero-advantage fraction, reward variance by group |
| Action quality | Malformed response rate, illegal parsed-action rate, effective-action rate |
| Resources | Elapsed time, iteration time, peak GPU memory, checkpoint step |

Do not substitute the average reward over expanded turn rows for the average episode return. The earlier run expanded four episodes into 32 turn-level records; those are not 32 independent episodes. If total loss is unavailable, label the actual logged loss components rather than inventing a combined number.

**Deliver to Prakeerth:** metrics, definitions, configurations, and checkpoint identifiers after the first short continuous run. Curves are still required if performance remains flat.

### A5 Train and select the final model

Use the shared evaluation protocol from B2. Run validation periodically and select the checkpoint using a rule fixed beforehand, such as highest validation win rate with earlier checkpoint as the tie-breaker. Keep the final test set untouched until selection is complete.

For the improvement bonus, compare the original and trained 0.5B model with the same maps, prompt, context, precision, action parsing, and decoding settings. A larger model or a different prompt is an additional baseline, not evidence that RL improved the original model.

If compute permits, repeat the chosen training configuration with three training seeds. If only one run is feasible, report it as a pilot and do not claim robust improvement. Prakeerth will report paired outcomes and uncertainty across distinct test maps.

**Deliver:** final adapter, intermediate adapters as useful, training configuration, selection rule, full learning curves, and observed successes/failures.

### A6 Deliver a small model handoff package

Transfer model files through an agreed Drive folder or another explicit file-sharing channel, not the Git repository. Keep generated results and weights ignored.

| Item | Purpose |
| --- | --- |
| Adapter configuration and nonempty adapter weights | Load the trained policy |
| Model manifest | Base-model and tokenizer revisions, adapter step, LoRA settings, numerical precision |
| Evaluation protocol | Prompt, history construction, response format, decoding, action validation |
| Run metrics and configuration | Plot curves and reproduce comparisons |
| Fixed prompt fixtures | Check local/Colab input and output consistency |
| File checksums | Verify transferred model files are intact |

Prakeerth should acknowledge a successful reload and one completed episode before you start a larger training campaign.

### A7 Run an additional environment for the environment bonus

After the core maze workflow works, select one genuinely different environment. Prefer a small existing environment in the pinned RAGEN checkout, such as FrozenLake if that version supports it; otherwise use an available lightweight environment or implement the minimal adapter. Inspect that checkout's examples rather than assuming current upstream commands match the local version.

YC owns registration/configuration and a short RL run. Prakeerth owns base-model evaluation, the corresponding scripted baseline, and the results table. Share the environment rules and observation format.

**Acceptance:** complete original-model episodes, a completed short training run with metrics, and post-training episodes. An import test alone is insufficient. Additional maze layouts are task variants, not automatically separate environments. Report returns separately when reward scales differ.

### A8 Package Strategic Blocking as a reusable RAGEN environment

The existing standalone adapter is a strong starting point for the new-environment bonus.

1. Add the environment using the extension structure expected by the pinned RAGEN version.
2. Include registration, configuration, a minimal example, and setup instructions.
3. Keep opponent tie-breaking, turn order, illegal-action handling, reachability checks, and terminal rewards explicit.
4. Add checks for reset/step behavior, terminal B-only turns, timeout boundaries, legal blocking, and new-layout loading.
5. Have Prakeerth independently run the environment and replay fixtures from Colab.
6. Prepare contribution-ready changes on a branch or fork. Confirm with the instructor whether integration in the project/fork suffices or whether an upstream pull request is expected.

Upstream submission is a later explicit action; this plan does not commit, push, publish, or submit a PR. An accepted upstream contribution cannot be promised.

**Deliver:** reusable environment implementation, tests, example configuration, English documentation, and a contribution summary.

### A9 Coordinate milestones and the final submission

| Milestone | YC work | Prakeerth work | Exit condition |
| --- | --- | --- | --- |
| M1 Independent setup | A1 and A2 | B1 and baseline notebook skeleton | Local training runs; Colab completes one base-model episode |
| M2 Shared protocol | Export fixtures; review maps | B2–B4 | Same rules and prompts; baseline results saved; adapter reload works |
| M3 Core experiment | A3–A5 | B5–B7 as artifacts arrive | Multi-iteration curves and actual model cases |
| M4 Bonus experiments | A7–A8 and controlled training ablations | B8–B9 | Evidence recorded for every attempted bonus |
| M5 Submission | Training methods, integration, limitations | Figures, cases, evaluation methods | Joint report, presentation, and proposal slides |

Complete the required outcomes before expanding bonus experiments. All bonus categories remain in scope; if time prevents a category from being completed, report it as incomplete rather than claiming credit.

For the report, YC writes the environment, RAGEN integration, training algorithm/configuration, compute setup, and training limitations. Prakeerth writes evaluation, plots, cases, and comparisons. Both review conclusions and maintain a task/commit/experiment contribution log.

For the presentation, prioritize interesting observed behavior and supported findings. Reserve 1–2 slides for the final-project proposal: the agent to build, user/task, available data resources, intended evaluation, and feasibility. Separate pilot results from future plans.

## Part 2 Prakeerth Google Colab Evaluation and Case Studies

### B0 Understand your role and start without waiting for training

You will call the existing Qwen model to play the maze, record its behavior, and compare it with YC's trained version. Transformers is only the loading/inference library. PEFT attaches YC's trained LoRA weights to the original model. Your main Colab notebook does not need Ray, vLLM, FSDP, or the complete RAGEN training stack.

You can immediately implement the evaluation notebook, run scripted baselines, evaluate the original model, construct maps, and prepare case visualizations. Trained-model evaluation begins when YC supplies a valid adapter.

The experiment loop is: render the current maze observation, call Qwen, parse the selected action, execute it in the shared environment, save the transition, and repeat until termination.

### B1 Build the Colab notebook step by step

Create `notebooks/evaluate_colab.ipynb` in your project branch. This notebook and the shared evaluation helpers below are proposed deliverables; they do not already exist in the inspected repository.

#### Cell 1 Select and inspect the runtime

Open a new [Google Colab notebook](https://colab.research.google.com/). Select a GPU runtime using Runtime, Change runtime type, then run:

~~~python
import sys
import torch

print(sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
~~~

If no GPU is available, work on maps, scripted policies, plots, and trace rendering on CPU. Do not make a larger model necessary for completing the main work.

Colab does not guarantee a particular GPU or uninterrupted runtime. Save completed results frequently; files in the temporary runtime are not durable. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

#### Cell 2 Mount persistent output storage

~~~python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
output_root = Path("/content/drive/MyDrive/strategic-blocking-pilot")
output_root.mkdir(parents=True, exist_ok=True)
~~~

Use local `/content` storage for execution and temporary model loading. Save finished episode records, manifests, plots, and notebook outputs to Drive. Copy an adapter from Drive to local storage once before loading it. Do not save the model cache after every action.

#### Cell 3 Obtain the project and select a shared revision

For a fresh runtime:

~~~bash
!git clone https://github.com/YC-Qiu/strategic-blocking-llm-agents.git /content/strategic-blocking-llm-agents
%cd /content/strategic-blocking-llm-agents
!git rev-parse HEAD
!python -m unittest -v
~~~

If the repository is private, obtain access or upload a source ZIP supplied by YC. Do not paste access tokens into notebook cells or outputs. Before a recorded experiment, check out the agreed project commit; both teammates must record its exact hash. Confirm that YC has made the required source revision available rather than assuming local changes are already on GitHub.

**Acceptance:** all 10 existing tests pass before adding new features.

#### Cell 4 Install the inference dependencies

Candidate bootstrap command for this plan:

~~~python
%pip install -q "transformers==4.56.1" accelerate peft safetensors pandas matplotlib pyyaml
~~~

The Transformers version matches the recorded local workflow. The full Colab dependency combination still needs a clean-runtime smoke test. Keep Colab's installed CUDA-enabled PyTorch initially. If imports require a session restart, restart, rerun setup, and check versions again.

After the model and adapter both load successfully, record all package versions and replace this bootstrap command with a tested pinned requirements file. Do not install the entire local RAGEN dependency stack just to perform inference.

#### Cell 5 Load the original Qwen model

The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) documents loading this existing model with Transformers. This is a setup smoke-test snippet, not the finished experiment runner:

~~~python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
if device == "cuda":
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    attn_implementation="sdpa",
).to(device).eval()

messages = [
    {"role": "user", "content": "Reply with exactly: <answer>wait</answer>"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
~~~

Record the resolved model/tokenizer revision and pin it for experiments. The smoke-test instruction above is only a loader check; actual maze evaluation must use the shared task prompt. Decode newly generated tokens only, so prompt text is not accidentally parsed as the action.

#### Cell 6 Exercise the existing environment without an LLM

~~~python
from ragen_maze import StrategicBlockingConfig, StrategicBlockingEnv

env = StrategicBlockingEnv(StrategicBlockingConfig(layout="slides"))
observation = env.reset(seed=0, mode="val")
print(observation)

observation, reward, done, info = env.step("wait")
print(observation)
print(reward, done, info)
env.close()
~~~

The current adapter accepts `slides` and `costly` only. Custom maps and the separate timing fixture require an interface extension; changing the seed does not create a new maze.

Reset already advances B. Step applies A and advances B again if needed. Do not manually move B a second time in the evaluation loop.

### B2 Implement a shared evaluation protocol

Coordinate with YC before collecting the main comparison. Put shared helpers in the repository and keep notebook cells focused on setup, execution, and presentation.

Proposed files:

| File | Responsibility |
| --- | --- |
| `evaluation/protocol.json` | Versioned prompt, decoding, context, and action rules |
| `evaluation/runner.py` | Episode loop and result recording |
| `evaluation/policies.py` | Scripted baselines and model policy interface |
| `evaluation/plots.py` | Curves and comparison plots |
| `evaluation/cases.py` | Trace selection and trajectory rendering |
| `notebooks/evaluate_colab.ipynb` | Colab entry point |
| `docs/evaluation.md` | Reproduction instructions and metric definitions |

Agree on these details:

- The exact instruction, tokenizer chat template, special tokens, and observation text.
- The precise construction of the two-turn history window. Inspect RAGEN's actual serialized prompts; do not guess whether “two turns” means two messages or two interaction pairs.
- One action in `<answer>...</answer>`, with direction where required.
- A 32-round limit and terminal win/loss/timeout rewards.
- Greedy decoding for the primary comparison, with the same output-token limit for base and trained models. Use the current 128-token action-only limit unless a documented protocol revision changes it.
- Matching precision and model revision for base/trained evaluation within Colab.
- Separate handling and metrics for malformed responses and parsed but illegal actions.

The current adapter converts a parsed illegal action to a turn-consuming wait. RAGEN may intercept a malformed response before the adapter is called. YC must identify the manager's actual behavior. Reproduce it or jointly define a revised evaluation-only wrapper for both models; label revised results separately from the original RAGEN baseline.

Never silently substitute a strategically helpful legal action for an invalid model response. Do not add retries or action masking to only one model.

**Consistency checks:** replay the same scripted action sequence locally and in Colab and compare states, rewards, and termination. Compare serialized model prompts and token IDs on fixed fixtures. Small backend numerical differences may change generation, so rerun both base and trained models through the same Colab evaluator for the main table rather than combining incompatible backends.

#### Episode records

Save one JSONL record per completed episode with run ID, model/adapter ID, checkpoint step, map ID/hash, split, seed, prompt version, decoding parameters, final outcome, terminal return, and action counts.

Within each episode retain every observation, supplied messages, raw model response, parsed action, executed action, validation status, reward, and next state. Mark crashes or interrupted episodes as incomplete; do not count them as losses or silently omit them from reporting. Resume by episode key to prevent duplicates.

**Acceptance:** the original Qwen completes episodes on both existing layouts; records can replay the actions and reproduce the final outcome.

### B3 Build maps and freeze the evaluation splits

First reproduce the existing slide and costly-detour cases. Add the timing fixture after extending the shared adapter. Use these as diagnostic demonstrations, not held-out evidence if they were used for training or tuning.

For the expanded experiment, start with a proposed 20 training, 10 validation, and 20 test maps if the available time permits. Prefer fewer carefully checked maps over a large unreliable generator.

Include:

1. Direct movement is sufficient and blocking wastes time.
2. A short detour and legal block can change a loss into a win.
3. Placement timing changes the opponent's detour cost.
4. Difficult cases where neither the tested direct policy nor the tested block heuristic wins.

Check reachability, legal placements, turn order, and scripted results. Label strategic categories based on tested policy outcomes; do not claim a global optimum unless an exact solver establishes it. The existing timing example compares travel costs; both scripted policies win, so it does not by itself prove timing determines the winner.

Assign stable IDs and hashes. Avoid identical maps and simple renamed/symmetric copies across splits. Give YC the training/validation maps. Freeze the test suite before tuning and reserve final test results for the chosen model.

**Deliver:** map files, split manifest, validation checks, and per-map scripted reference outcomes.

### B4 Run the baseline suite

| Baseline | Definition | Purpose |
| --- | --- | --- |
| Direct shortest path | A follows its current BFS path to its goal and never blocks | Tests whether blocking is needed |
| Immediate legal blocking heuristic | At the first decision with a legal block, choose the block maximizing the increase in B's current shortest-path distance; use fixed directional tie-breaking, then follow A's shortest path | Tests whether indiscriminate early blocking is sufficient |
| Existing scripted detour | Replay the authored strategy on its applicable demo map | Positive environment reference, not a general learned policy |
| Original Qwen 0.5B | Shared action-only prompt, without RL adapter | Primary baseline for measuring RL improvement |
| Original Qwen with brief rationale | Separate explanation-enabled prompt | Diagnostic prompt baseline |
| Larger Qwen model | Only if Colab resources allow | Model-size bonus comparison, not proof of RL improvement |

For the blocking heuristic, consider only environment-legal placements. If none is available, move toward A's goal and try again at the next decision. Log the selected cell and the distance estimates.

Run each deterministic policy once per distinct map. Repeating identical greedy episodes with different seed labels does not create independent evidence. If adding sampled decoding, use a declared seed list and record the decoding settings.

Report win rate with numerator/denominator, mean terminal return, timeout rate, malformed-response rate, illegal-action rate, and block-use rate. Compare step counts among comparable outcomes; a quick loss should not appear more efficient than a longer win.

### B5 Load and evaluate YC's adapter

Wait for A2's validated export, not for proof that training succeeded. The [PEFT documentation](https://huggingface.co/docs/peft/en/quicktour) describes attaching adapter weights to a base model with `PeftModel.from_pretrained`.

In a fresh model-loading cell:

~~~python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    attn_implementation="sdpa",
).to(device)

trained_model = PeftModel.from_pretrained(
    base_model,
    "/content/adapter",
    is_trainable=False,
).eval()
~~~

Copy the supplied adapter into `/content/adapter` first. Use the exact base/tokenizer revision in YC's manifest. Release the earlier model if memory is limited; avoid keeping several unnecessary copies on the GPU. Verify checksums, adapter configuration, nonempty weights, and fixed-prompt fixtures before evaluating.

Evaluate original and trained models using B2's identical protocol. Select checkpoints on validation data, then run the selected model on the frozen test set. Report wins gained, wins lost, and unchanged outcomes by map. If available, include results across training seeds and uncertainty over distinct maps; small suites must be presented as pilot evidence.

If export remains blocked, finish baseline evaluation, plots, and cases while YC repairs it. Do not label an original-model run as a trained-model result.

### B6 Produce cases about decisions and explanations

Create at least four informative cases, using actual observed behavior:

| Case | Question |
| --- | --- |
| Useful detour | Does the model leave its shortest route when blocking can pay off? |
| Costly blocking | Does it avoid blocking when direct movement is better? |
| Timing | Does it recognize that placement time affects B's route or travel cost? |
| Failure or mismatch | Does its brief explanation contradict the legal actions, map, or chosen action? |

If no trained model succeeds, use a clearly labeled scripted success beside the actual model failure. Do not invent a successful model trace.

Keep action-only evaluation as the primary protocol. Add a separate diagnostic condition asking for a brief decision explanation before the action, for example a short `<rationale>...</rationale>` followed by `<answer>...</answer>`. Update the parser deliberately, allocate the same rationale token budget to base and trained models, and record truncation. Do not assume that changing `enable_think` alone produces a compatible protocol.

These are model-generated explanations, not verified access to an internal reasoning process. Check factual correctness and agreement with actions. If the model was trained action-only, explain that the rationale condition changes its evaluation prompt.

For every selected case show the initial map, a few decisive states, exact model text, requested/executed actions, final outcome, and a short interpretation. Include the selection rule, such as the first observed valid block, one representative loop, or one success-to-failure change, so the cases are not presented as a representative success rate.

**Deliver:** four or more case panels and an English analysis connecting them to the assignment's thinking-process requirement.

### B7 Plot curves and investigate training findings

Use YC's raw metrics to produce:

1. Policy-gradient loss versus training iteration.
2. Mean episode return versus iteration.
3. Training success rate and, when available, fixed validation success rate.
4. Entropy and the fraction of informative GRPO groups.
5. Optional resource plots: iteration time and GPU memory.

Show raw values; any smoothed line must identify its smoothing window. Keep separate runs distinct and label seed, curriculum changes, and checkpoint points. Include wall time as an additional axis or plot when useful. Do not manufacture a continuous curve by joining independent fresh one-step runs.

A flat zero policy loss or flat -1 reward curve is a valid observed result. Explain its meaning. Policy loss is not a supervised error metric and need not decrease monotonically for behavior to improve.

For the interesting-findings bonus, prioritize controlled experiments:

| Finding to investigate | Comparison | Evidence |
| --- | --- | --- |
| Zero advantages despite changed weights | Matched short runs with entropy coefficient zero versus the current value, with other settings and weight decay recorded | Rewards, advantages, loss terms, gradients, weight differences |
| Curriculum and reward diversity | Fixed difficult tasks versus easier-to-harder training under a declared rollout budget | Informative-group fraction and held-out evaluation |
| Explanations and actions | Action-only versus brief-rationale evaluation | Outcome differences, factual errors, explanation/action mismatch |
| Blocking timing | Immediate versus delayed scripted placements, then model choices | Opponent travel cost and final race outcome |

YC executes training ablations; you define plots and analyze their output. If all rewards remain identical, zero advantages are expected and should not be misreported as a training-code failure by themselves.

### B8 Cover the additional environment bonus

Coordinate with A7 on one separate environment. Use the same Qwen inference wrapper but the environment's own observation/action rules. Implement an original-model baseline and a simple suitable scripted or random policy. Run complete episodes, then evaluate the environment-specific trained adapter if supplied.

Keep environment identities and reward scales separate. The extra environment does not establish generalization of the maze-trained policy unless that transfer experiment is explicitly performed. Record setup time and failures as well as final results.

### B9 Independently validate the new environment contribution

From a fresh Colab session, follow YC's environment setup instructions, run tests and scripted replays, and load the agreed example configuration or equivalent documented wrapper. Check that all necessary source files and instructions are actually available outside YC's machine.

Provide reproducible bug reports with the project revision, map, action sequence, expected behavior, and actual behavior. This validates the new environment itself. Full RAGEN training integration remains YC's responsibility; pure Colab wrapper inference does not independently verify distributed training.

**Deliver:** a short reproducibility checklist and fixes needed before A8's contribution package is finalized.

### B10 Assemble the report and presentation evidence

Use the following report outline:

1. Research question and task motivation.
2. Environment, opponent, and reward design.
3. RAGEN training setup and model configuration.
4. Evaluation protocol, map splits, and baselines.
5. Loss/reward curves and quantitative results.
6. Case studies with actual model explanations.
7. Bonus experiments and what each demonstrates.
8. Limitations, negative findings, reproducibility, and individual contributions.
9. Final-project direction where required by the course.

Prepare a presentation that leads with a compelling observed case, then explains the environment, training evidence, curves, baseline comparison, and limitations. End with the required 1–2 proposal slides describing the future agent and data resources.

The submission checklist is:

- [ ] A completed, reproducible RL training run is documented.
- [ ] Multi-iteration loss and reward curves are included.
- [ ] Actual model behavior and brief-explanation cases are shown.
- [ ] Interesting RL findings have supporting experiments.
- [ ] Additional baselines and one additional environment have completed runs.
- [ ] Any claimed improvement uses a fair base-versus-trained comparison.
- [ ] The new environment has reusable integration, tests, and documentation.
- [ ] Report and presentation are complete, with 1–2 final-proposal slides.
- [ ] Each teammate's contributions are recorded.
- [ ] Unfinished bonus attempts and negative results are labeled honestly.

Keep all code comments, notebook explanations, figure labels, reports, and slides in English. Do not commit raw results or weights; commit reviewed compact summaries and figures. This plan creates no training job, commit, push, or public release.

### Reference material

- Project source: [Strategic Blocking LLM Agents](https://github.com/YC-Qiu/strategic-blocking-llm-agents).
- Training framework: [RAGEN](https://github.com/mll-lab-nu/RAGEN); use the pinned local version for actual commands.
- Model loading: [Qwen2.5 0.5B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Adapter loading: [PEFT quicktour](https://huggingface.co/docs/peft/en/quicktour).
- Hosted runtime behavior: [Colab FAQ](https://research.google.com/colaboratory/faq.html).
- Local project instructions: `RAGEN.md`, `docs/local-training.md`, and `scripts/run_ragen.sh`.
