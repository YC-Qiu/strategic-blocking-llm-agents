# Submission status and evidence

This status reflects the audited Colab v2 output and native Bandit run
`train-bandit-0WF04q1N`. It supersedes older pending-work notes in historical
runbooks. Available experimental evidence is distinct from a finished report or
presentation. Bonus points remain subject to instructor assessment.

## Required outcomes and bonuses

| Item | Current evidence | Remaining work or limit |
|---|---|---|
| Run an RL framework | RAGEN rollout/update/checkpoint workflow completed for Strategic Blocking and native Bandit | Clean-install replication remains unverified |
| Show loss and reward curves | Maze curriculum figures and Bandit loss/reward/choice curves | Select and explain figures in final report and slides |
| Visual case study, especially thinking | Interactive local blocking replay; 46 local and 40 Colab winning traces fail after first-block removal with remaining actions fixed | Four panels plus a completed 72-request explanation diagnostic and reviewed failure examples; no verified internal thinking process |
| Interesting RL findings | Sparse-reward zero advantages, prompt/protocol sensitivity, layout-dependent transfer, Bandit payout versus decision quality | State observed findings separately from untested causal explanations |
| More environments / baselines | Separate native Bandit training, base/adapter evaluations, random/first-position baselines and privileged controls | Bandit improvement not demonstrated; extra maze layouts are not separate environments |
| Improve the baseline | Local blocking 2% base versus 92% final; Colab wide map 2% versus 80% | One training chain; mirror/navigation failures prevent a broad transfer claim |
| Add a new environment | Custom Strategic Blocking adapter, configuration, tests, documentation and live RAGEN registry audit | Local extension only; no upstream merge claimed |

## Original plan: A tasks

| Task | Status | Evidence and boundary |
|---|---|---|
| A1 Local workflow | Working on existing installation | Multiple actual training runs; clean installation not independently repeated |
| A2 Adapter export | Working | Checkpoint/export checks and actual Colab adapter loading; common cached base, not exact full-checkpoint replay |
| A3 Informative reward | Achieved in the final pilots | Five nonzero-advantage updates in the final blocking and Bandit runs; no controlled causal curriculum ablation |
| A4 Training curves | Available | Named loss and actual reward metrics; five updates per final stage |
| A5 Trained model and evaluation | Pilot evidence available | Strong local gains and limited Colab transfer; independent training seeds remain absent |
| A6 Handoff | Completed for v2 evaluation | Hashed adapters/protocol loaded on T4 and 1,200 episodes completed |
| A7 Additional environment | Completed pilot | Native Bandit, 80 training episodes, 200 test instances per model, four controls |
| A8 Environment contribution | Local integration verified | Runtime registry and four scripted replay checks; clean install/upstream acceptance not claimed |
| A9 Final submission | Pending assembly | Report, slides, proposal, contribution record and final Git review |

## Original plan: B tasks

| Task | Status | Evidence and boundary |
|---|---|---|
| B1 Colab setup | Working | T4 generation and PEFT loading; dependency adjustment recorded |
| B2 Evaluation protocol | v2 completed with limits | Repaired prompt/parser/history and gates; original smoke files are not in the downloaded full-run folder |
| B3 Frozen maps | Evaluated with disclosure | Eight frozen maps; v2 followed inspection of v1 failures, not a newly blind test |
| B4 Baselines | Partially covered | Base LLM, scripted/rule controls and separate Bandit controls; the full proposed immediate-block/rationale/larger-model matrix was not established |
| B5 Adapter evaluation | Completed v2 pilot | Base/before/after, 400 episodes each; all downloaded trajectories replayed |
| B6 Case studies and explanations | Behavior panels and explanation diagnostic completed | All 36 explanation responses inspected; severe format failures and factual/action contradictions documented; no claim of faithful internal reasoning |
| B7 Curves and findings | Core evidence available | Existing figures and audits; proposed controlled ablations remain incomplete |
| B8 Additional environment comparison | Completed locally | Native RAGEN Bandit evaluations ran on YC's Ubuntu, not Colab; preserve actual attribution |
| B9 Independent environment validation | Partial | Colab uses the environment and outputs replay correctly; a separate clean RAGEN install/checklist is not supplied |
| B10 Report and presentation | Pending | Assemble final artifacts and the required proposal slides |

Do not label every original A/B acceptance criterion complete. The pilot covers
the main experiment and has evidence for each bonus category, but several
optional or ambitious plan items were not performed.

## Submission sequence

1. Update repository documentation with the audited tables, JSON summaries and
   selected curves. Preserve raw runs, failed v1 output and weights outside Git.
2. Review the four prepared [case panels](case-studies.md): a learned local blocking success; a Colab wide-map
   success and fixed-action intervention; a mirrored-map failure; and the scripted
   costly-detour control. Label scripted versus learned cases explicitly. Select
   cases by a declared rule, such as the first qualifying episode, and show exact
   model output, executed actions and decisive states. The four behavior panels and the separate
   [brief-explanation diagnostic](rationale-diagnostic-results.md) are complete.
   Present the observed output failures honestly; no internal reasoning should
   be inferred from these artifacts.
3. Write the report using methods, curves, matched results, cases, bonus evidence,
   limitations and individual contributions. Do not combine different rewards or
   independently trained models into one aggregate learning curve.
4. Build presentation slides around the observed cases and findings. Include
   1-2 proposal slides describing the intended future agent and data resources.
   A possible extension is a generalizing strategic-blocking agent trained on
   procedurally generated solvable mazes with held-out layout families and fixed
   opponent variants. This is a proposal direction, not work already completed.
5. Review tracked changes, run the existing tests and whitespace check, then
   explicitly decide on commit/push. No publication is performed by this update.

Multi-seed training and a new held-out suite would strengthen research claims,
but they need not block packaging this bounded pilot. If training resumes after
examining these results, treat the eight Colab maps as development evidence and
freeze a fresh evaluation suite before training.

## Evidence index

- [Local final results and curves](final-pilot-results.md)
- [Local interactive case](blocking-case.html)
- [Colab v2 audited results](colab-v2-results.md) and [machine-readable audit](colab-v2-results.json)
- [Bandit results and curve](bandit-results.md) and [record audit](bandit-results-audit.json)
- [Environment contribution](environment-contribution.md) and [runtime check](environment-runtime-check.json)
- [Explanation diagnostic results](rationale-diagnostic-results.md), [reviewed rows](rationale-semantic-review.csv), and [example cards](rationale-diagnostic-examples.html)
- [Full original execution plan](../Strategic_Blocking_RL_Pilot_Plan.md)

The supplied assignment mentions April 28 and Week 5. Confirm the applicable
course deadline rather than treating that undated schedule as a current date.
