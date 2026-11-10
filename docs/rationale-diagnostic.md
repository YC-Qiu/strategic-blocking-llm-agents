# Brief explanation diagnostic

## Purpose and current status

The existing policy outputs actions only. Its recorded trajectories cannot be
described as an explicit thinking process. This follow-up elicits a short
decision explanation before a fresh action and checks it against the displayed
map and the selected action. It is an exploratory prompt intervention, not
access to internal reasoning or a retrospective account of an earlier choice.

Run `rationale-v1-j4rtnkvx` completed all 72 requests, and all 36 explanation
responses have been reviewed against state facts. Base produced 3/18 structurally
valid explanation responses (only one also legal); the final adapter produced
0/18. Preserve this negative result and its condition-specific parsing limits.
See [the audited results](rationale-diagnostic-results.md),
[reviewed rows](rationale-semantic-review.csv), and
[actual output examples](rationale-diagnostic-examples.html). The commands and
rubric below describe the original reproducible protocol, not unfinished work.

## Fixed design

| Dimension | Choice |
|---|---|
| States | Local pre-block state; wide initial; mirrored wide pre-block state; costly-detour initial; direct initial; mirrored direct initial |
| Policies | Original base and final blocking adapter from `train-blocking-N6WUl30K` |
| Conditions | Action-only control; brief explanation followed by an action |
| Samples | Three request seeds per state, policy, and condition: 72 requests total |
| Backend | Transformers/PEFT, FP16, SDPA, batch size one; same base and final adapter as Colab v2 |
| Budget | 192 new tokens for each condition; context plus generation at most 2,048 tokens |
| Sampling | Temperature 1, top-p 1, top-k 0; matched request seeds across policies and conditions |
| State history | Fresh single-state snapshot, with accurate environment round and remaining actions; no selected historical model answers supplied |
| Training | None; the final adapter is frozen |

These states were chosen after examining earlier results, so they are not held
out. The local and mirrored pre-block states are reproduced by executing `move up`
from their respective resets; the other four are initial A decisions after B moves. The control prompt's turn
number restarts at one because every request is independent. This diagnostic
does not recreate full historical contexts or estimate episode win rates.

Both conditions use the same total token budget. The explanation condition
changes instructions and assistant prefill from `<answer>` to `<rationale>`.
It requires one or two sentences followed by exactly one tagged action. Its
strict parser rejects incomplete tags, empty explanations, extra text and
multiple actions. The action-only parser retains the captured RAGEN behavior.
There is no bare-action fallback or automatic retry that removes failures.
The different output requirements are part of the intervention; format rates
are not presented as a pure measure of reasoning ability.

## Run on the existing Ubuntu environment

```bash
conda activate ragen
cd /mnt/d/Git/strategic-blocking-llm-agents
python -m unittest test_rationale_diagnostic -q
python scripts/run_rationale_diagnostic.py --check-only
python scripts/run_rationale_diagnostic.py
```

The default adapter is `results/paired-eval-trum2mve/verified_adapter`. Its
configuration and weight hashes must match the recorded final blocking policy.
The base model and tokenizer load from the pinned local cache; the script does
not install packages or download another model. Stop any concurrent GPU training
before running this separate inference job.

The runner prints a new `results/rationale-diagnostic/rationale-v1-*` directory.
It saves the protocol, dependency versions, source hashes, tokenizer identity,
every complete prompt and response, prompt/completion token IDs, request seeds,
format validity, action legality, and token-budget exhaustion. Runtime errors
leave `FAILED.json` and any completed responses. A completed generation run has
`COMPLETE_GENERATION`, which explicitly does not mark semantic review complete.

No win rate is computed. Generated actions are classified for legality; these
independent requests do not drive full environment episodes. The factual review
reference separately enumerates every legal action's effect from each snapshot.

## Manual review rubric

Review all 36 explanation-condition requests, including failures, using
`responses.jsonl`, `state-facts.json`, and `manual-review.csv`. Do not inspect only
the most persuasive explanation. Preserve the raw JSONL and put judgments in
the CSV. Action-only responses have no explanation to assess.

| Review column | Labels | Evidence to use |
|---|---|---|
| `factual_correctness` | correct / incorrect / mixed / underspecified / not_assessable | Positions, goals, legal directions, occupancy, obstacle availability, and shortest-path changes |
| `rationale_action_agreement` | agrees / contradicts / underspecified / not_assessable | The action explicitly described in the explanation versus the tagged action |
| `strategic_justification` | supported / unsupported / uncertain / not_assessable | Whether the cited mechanism is supported by this state and the action-effect reference |
| `evidence_and_notes` | Short quotation plus concrete evidence | Quote the relevant generated words and cite the state fact or action effect; separate missing evidence from a false statement |

The state facts distinguish shortest distances **after A acts but before B
moves** from the next observation **after B moves**. Use the former for a claim
that a block lengthens B's path; mixing these moments can give a wrong judgment.
Distances alone do not establish which player wins. A claim that an action
guarantees victory is uncertain unless supported by a separate full trajectory
or planning check. A legal action is not automatically strategically good.

Use `not_assessable` for missing or malformed explanations. If the response
repeats an action without a reason, record insufficient strategic justification
rather than rewarding its length or confidence. A factual explanation may still
describe a losing choice; action agreement and strategic quality are separate.

## Reporting the result

Report format-valid and legal-action counts out of all 18 requests per
policy/condition, plus truncation counts. Explanation quality uses all 18
explanation requests per policy as the primary denominator; report unassessable
responses explicitly and identify any conditional denominator.

The runner also counts action changes between conditions only where both paired
outputs are format-valid and legal. Always show that eligible-pair denominator.
A shared random seed does not make the token trajectories identical across
different prompts, and the three sampling seeds are not independent training
seeds. No causal claim about the original action-only model's internal process
follows from this experiment.

After reviewing all responses, choose examples by a declared rule: first fully
supported explanation, first factual contradiction, first explanation/action
contradiction, and first unassessable response in recorded order, if each exists.
If a category is absent, state that it is absent; do not invent an example.

The report can conclude whether the small model produces checkable explanations,
whether they agree with its actions, and whether asking for explanations changes
its choices on these six states. Even a mostly malformed or generic response
set is a valid negative diagnostic finding. Do not rename this result a verified
chain of thought or merge it into the primary Colab/local policy scores.

See [the four behavior cases](case-studies.md) or open
[the offline case panels](case-studies.html). Those cases are already backed by
recorded actions and outcomes; explanation diagnostics remain a separate phase.
