# Brief-explanation diagnostic: results and review

Run: `rationale-v1-j4rtnkvx`. All 72 independent requests completed on the local RTX 3070 using the pinned base and final blocking adapter. This was inference only, with six selected states, two policies, two prompt conditions, and three request seeds. It is not a win-rate evaluation or a record of internal reasoning.

## Main finding

The action-only interface remained mostly usable, but the new explanation-plus-action interface failed frequently for both policies. Readable output fragments also contain factual errors and action contradictions. This demonstrates prompt/interface fragility in this exploratory setting; it does not establish that RL reduced reasoning ability or that the model cannot reason.

| Policy | Condition | Requests | Format valid | Format-valid and legal | Token budget exhausted |
|---|---|---:|---:|---:|---:|
| base | action_only | 18 | 18/18 | 15/18 | 0/18 |
| base | brief_rationale | 18 | 3/18 | 1/18 | 6/18 |
| after | action_only | 18 | 18/18 | 17/18 | 0/18 |
| after | brief_rationale | 18 | 0/18 | 0/18 | 1/18 |

All explanation-condition outputs total 36 requests: 3 have valid structure and only 1 also has a legal action. Seven exhausted the 192-token budget (6 base, 1 after). No failure was removed or retried. This is diagnostic completion, not successful explanation generation.

The two conditions use their predeclared parsers. Action-only accepts the first tagged answer and may ignore trailing text; explanation mode requires a complete rationale/answer pair with no extra text. Consequently these format counts measure compliance with different requested interfaces, not a parser-independent reasoning score. Zero valid after-policy explanation outputs does not mean zero recognizable legal-action strings in its raw text.

## Semantic review

All 36 explanation requests were inspected against the exact state and legal action-effect tables. This is an assistant-assisted analyst review, with no independent second rater. In the primary rubric, malformed outputs remain not assessable. Readable errors in those outputs are reported separately as qualitative examples; the frozen parser and primary denominators were not changed.

| Policy | Explanation requests | Structurally unassessable | Valid but factually/action-wise underspecified | Fully supported explanation identified |
|---|---:|---:|---:|---:|
| Base | 18 | 15 | 3 | 0 |
| After | 18 | 18 | 0 | 0 |

The three structurally valid base outputs were: a refusal with a paragraph in the answer field; the sole legal `move up` with a generic coordinate-based explanation; and `Move down.` with the ambiguous rationale 'B is moving next.' The last action is both incorrectly formatted as an action string and blocked by a wall even if punctuation were stripped. The next-player wording alone is not treated as a definite factual error because B does move after A if the race stays active.

No fully supported explanation was identified under this conservative rubric. This is not a measured 0% reasoning ability: almost every response lacks the required assessable structure. A separate readability-based semantic metric would require a new declared analysis and must not silently replace this primary result.

## Selected real examples

### 1. A legal action with an underspecified explanation

Record index 23 (zero-based); `base`, `wide_initial`, repetition 2.

> Move from (2, 3) to (3, 7).

```text
<answer>move up</answer>
```

The coordinates match the start and goal, but the explanation does not justify moving up. That move increases A's shortest remaining distance from 5 to 6. A later detour could still help; the recorded explanation supplies no such mechanism.

Selection: The only format-valid legal response in the explanation condition.

### 2. An incorrect goal coordinate

Record index 20 (zero-based); `base`, `local_pre_block`, repetition 2.

> Move 3 cells up to reach goal (3, 7). Opponent blocked.

This is the local pre-block state: A's goal is (3, 6), no obstacle has been placed, and each move is one cell. The quoted text is outside the rationale tag and the response is malformed; it is qualitative raw-output evidence only.

Selection: First explanation-condition response with an explicit wrong A-goal coordinate in file order.

### 3. The explanation and later action prose disagree

Record index 55 (zero-based); `after`, `local_pre_block`, repetition 1.

> You can choose "block right" to reduce the risk of getting stuck anymore.

```text
Your legal action is now: move right
```

Block right and move right are different actions. The response also says B can clear obstacles, but obstacles are permanent. This is a contradiction in readable raw text, not a comparison against an accepted executable answer: the answer tag is missing.

Selection: First after-policy response with an explicit named-action contradiction between its rationale and subsequent action prose.

### 4. An obstacle that does not exist

Record index 35 (zero-based); `base`, `direct_mirror_initial`, repetition 2.

> B has already placed an obstacle

The observation says Obstacle: None, and only A has a placement action. This is a direct state/rule contradiction in a malformed response. No parsed action is rescued from the surrounding text.

Selection: First response explicitly claiming B has already placed an obstacle.

### 5. Useful direction, unusable output protocol

Record index 68 (zero-based); `after`, `direct_initial`, repetition 2.

> Move right as it offers the fastest movement speed.

```text
<answer> move_right </answer>
```

Move right immediately wins in this state, so the prose points in a useful direction. However, movement speed does not vary by direction, and move_right is not a legal action string. Extra text also violates the format. This separates recognizable intent, factual support, and interface compliance.

Selection: First after-policy explanation response on the direct-win state that explicitly proposes the immediate winning direction.

## Comparison limits

Only one base sample pair has format-valid legal actions in both conditions; its action changes from wait to move up. There are zero eligible after-policy pairs. This is insufficient to assess a systematic effect of explanations on action selection. Do not report a 0% after-policy change rate when its denominator is zero.

The action-only snapshot controls also show limited spatial transfer: the after policy chooses the immediate winning move right in all three direct-right snapshots, but none of its three mirrored direct snapshots chooses the immediate winning move left. These are independent decisions, not full-episode losses. They use a new sampling protocol and fresh single-state context, so they do not replace the earlier 1,200-episode Colab evaluation.

Both conditions have the same total generation budget, but explanations consume additional tokens and require unfamiliar tags. The rationale prompt, output requirements, prefill, temperature, and snapshot context are potential contributors. This pilot does not isolate them. Base also fails heavily, so do not attribute the failure uniquely to RL training or claim that it erased reasoning.

## Course-report wording

We inspected actual policy trajectories and performed a separate 72-request explanation diagnostic. The trained policy produced mostly legal actions under its action-only interface, but neither policy reliably followed the new explanation-plus-action format. Readable explanations sometimes invented obstacles, confused control of the two runners, or contradicted their later action text. We therefore report observable behavior and explanation failure modes rather than claiming access to an internal thinking process.

This supplies an actual attempt to investigate the assignment's thinking-process question, including negative findings. It does not need to be rerun until a persuasive explanation appears. Keep the failed outputs and use the [four behavior cases](case-studies.md) alongside this result.

See [all reviewed rows](rationale-semantic-review.csv), [audit and example metadata](rationale-diagnostic-results.json), and the [original protocol](rationale-diagnostic.md). Raw prompts, tokens and outputs remain in the ignored run directory. Report and presentation assembly are still separate deliverables.
