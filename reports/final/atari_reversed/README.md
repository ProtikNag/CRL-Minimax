# Line 1: five-game Atari sequence, reversed order

Reversed order **SpaceInvaders → Boxing → Breakout → Pong → Q\*bert**. Seed 0,
greedy-100 evaluation, plain `impala_ac_multihead` net, identical task set,
thresholds, PPO hyperparameters and frame-matched budget for both methods.
Single seed, so no error bars.

Order sensitivity is reported in the paper as an observed phenomenon, with the
two orders' retention matrices as the evidence. Its root cause and any mitigation
are out of scope here, so the canonical order is **not** rebuilt in this folder;
`reports/order_sensitivity/` keeps that material.

## Status, 2026-09-18

**Complete.** All four runs have finished all five tasks, so every number in
this folder is measured and nothing is extrapolated.

| Method | Tasks finished | Source |
|---|---:|---|
| Min-Max (ours) | 5 / 5 | `data_live.json`, post-threshold-fix |
| CLEAR | 5 / 5 | `../../order_sensitivity/data.json` |
| CKA-RL | 5 / 5 | `data_live.json`, post-threshold-fix |
| CompoNet | 5 / 5 | `data_live.json`, post-threshold-fix |

Ours' final row against each game's joint ceiling.

| | SpaceInv | Boxing | Breakout | Pong | Q\*bert |
|---|---:|---:|---:|---:|---:|
| raw | 762.6 | 10.5 | 235.6 | 17.7 | 5195.2 |
| vs ceiling | 84% | **16%** | 83% | 86% | 122% |

Four of five games end above 83%. **Boxing is the casualty**, decaying
monotonically through the sequence at 68.7 → 59.8 → 42.5 → 10.5 and ending at
16% of its ceiling. No other method holds four games this high, and none of the
others learns the final task as well; ours' Q\*bert diagonal of 5195.2 clears
both the 4260 threshold and the 4750 local specialist.

Caveats that have to travel with any of these figures.

**CLEAR's run predates the threshold change**, which does not affect it. CLEAR
never reached its thresholds on the first four games, so it trained to its full
budget on them regardless. Only Q\*bert cleared its threshold, and Q\*bert is
the last task, so nothing earlier is affected. CLEAR does not need rerunning.

**The threshold fix landed hard.** Ours' SpaceInvaders went 588.5 → 1318.1 on
task 1, which is 146% of the joint ceiling. That is the single largest change
in this tier and it is why the old and new runs are never averaged together.

**Ours' Breakout shows the value-vs-score gap, then recovers from it.** The
local phase peaked at 396, the value-constraint shortfall reached ≈0, and the
greedy diagonal still fell to 77.6 during consolidation. The constraint is
satisfied on `V` while the score collapses. It then climbed back
non-monotonically across the two following phases, 77.6 → 144.5 → 235.6,
finishing at 83% of its ceiling. Both halves belong in the paper, the
value-versus-score mismatch as a limitation of constraining value, and the
recovery as evidence that consolidation reclaims a task the constraint had let
slip.

**Model selection is the final iterate, not the best checkpoint.** This run kept
the final global of each phase rather than its best checkpoint, because the
global trainer had no best-checkpoint facility when it ran. That has since been
added and verified, so future runs will differ. Worth disclosing, since Boxing's
decay is exactly the kind of thing a best-checkpoint rule would have caught.

**A forgotten task cannot be patched afterwards.** Post-hoc recovery of Boxing
from the final global was tried three ways and all three were discarded as
net-negative. Ten iterations of pure Boxing moved it +2.3 while costing Breakout
159 and Q\*bert 1648. A hundred iterations of pure Boxing bought +36 on Boxing
and lost 4424 on Q\*bert. A hundred iterations of a 40/60 Boxing-to-rest mix
gained 23.5 on Boxing for a net −0.81 in normalised terms. The final matrix
stands unchanged, and the ablation is worth reporting because it argues that
retention has to be maintained during the sequence rather than repaired at the
end.

### What the two finished baselines did

The two failures are opposite, and between them they frame the stability and
plasticity trade the method is for.

**CKA-RL forgot almost everything.** Its final row is Space Invaders 212.6,
Boxing −15.5, Breakout 6.7, Pong −21.0, Q\*bert 4420.5. Boxing and Pong finish
**below a random policy**. Only the last-learned game survives, which leaves
average performance at 0.01 against a ceiling of 1.00 and backward transfer at
−0.99, the worst of the four.

**CompoNet forgot nothing and stopped learning.** Its lower triangle is
constant by construction, since components freeze, so backward transfer is
exactly 0.00 on every task. The cost shows up on the diagonal: Breakout reached
only 99.2 against a threshold of 285, and Q\*bert scored **0.0**, never learned
at all. Average performance lands at 0.66, and the whole of the shortfall is the
one task it never learned.

Ours sits between the two. On the final row it holds four of five games above
83% where CKA-RL holds none and CompoNet holds three, and it is the only method
that both retains earlier tasks and learns the last one.

| | Backward transfer | Forgetting | Average, all | Average, before last |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | **−0.26** | 0.40 | **0.78** | 0.67 |
| CLEAR | −0.62 | 0.62 | 1.07 | 0.43 |
| CKA-RL | −0.99 | 0.99 | 0.01 | −0.25 |
| CompoNet | +0.00 | **0.00** | 0.66 | **0.83** |

Read the two averages together. CompoNet leads before the last task precisely
because frozen components cannot be overwritten, and its 0.66 overall is what
that costs. CLEAR's 1.07 is an artefact of a Q\*bert score 3.6× its ceiling
while everything before it decayed, which is why it falls to 0.43 once the last
task is excluded. Ours is the only column strong in both.

## Normalisation

Everything in this folder is on **one scale, `score / Joint ceiling`**. The
retention matrices always used it and `transfer_table` now does too, so a cell
in a matrix and the aggregate under it are the same quantity.

The random floor is **not** 0 on this scale. It is 16% on Space Invaders, 4% on
Q\*bert, near 0% on Boxing and Breakout, and **−100% on Pong**, whose random
policy scores −20.7 against a ceiling of 20.7. So CKA-RL's Pong cell at −101%
means *at random*, not far below it, and a reader has to be told that once.

## Figures

| Stem | What it shows |
|---|---|
| `forgetting_matrices` | Retention matrices, 2×2, all four methods |
| `final_scores` | Per-game score after the final task, five series |
| `transfer_table` | Four aggregate metrics per method, no per-task rows |

`backward_transfer_matrix` and `compute_cost` were dropped. Compute was measured
across different GPUs per method, so the comparison was never fair.

## Nothing is stood in for

Every cell in this folder is measured. Earlier drafts padded a run's unreached
tasks at ours' value, which was defensible while the baselines were the ones
still going but became untenable once ours was the run in flight, since the only
complete five-task run of ours available to pad from was the pre-threshold-fix
one, whose task 1 scored 588.5 against 1318.1. The padding was removed then and
is not needed now.

The machinery that handles partial runs is still in `make_figures.py` and is
driven by the data, so a future partial run renders correctly without an edit.
An unreached task is left absent rather than stood in for, the caption naming
which runs are short derives from the row counts rather than being typed, and
every aggregate in `transfer_table` is computed over the tasks its own run has
finished.

**The local-specialist reference is only partly refreshed.** The cached vector
belongs to the pre-threshold-fix run. Task 1 has no prior global to consolidate
against, so the specialist's score and that run's own task-1 diagonal are one
measurement rather than two, and in the cached run both read 588.5 to the
decimal. `final_scores` therefore takes task 1 from the live diagonal, 1318.1,
computed at render time rather than typed. The identity holds for no other task
(Boxing 98.9 against 96.85, Breakout 363.9 against 293.58, Q\*bert 4341.0
against 4270.25), so **Boxing, Breakout, Pong and Q\*bert still carry pre-fix
specialist values** and can only be corrected by re-running the specialists.

**The score axis runs below zero where the data does.** CKA-RL finishes Boxing
at −15.5 and Pong at −21.0, both under the random floor. A zero-pinned axis drew
those two bars with no height at all, which read as *not reached* against a
caption saying exactly that. The floor now follows the data.

## Forward transfer is unresolved

Every forward-transfer figure in this project carries `*`. Two separate reasons:

- On Atari it is **not measurable** from these runs. Each task has its own head,
  untrained until that task arrives, and no task is evaluated before training.
- The from-scratch baseline that the AUC form needs is being recomputed, so the
  GridWorld numbers that *are* measurable are provisional too.

## Rebuild

```bash
python reports/final/atari_reversed/make_figures.py
```

Reads `data_live.json` for the three live methods and the cached transcription
for CLEAR. Re-transcribe `data_live.json` as the run advances; the figures and
every count in their captions follow the data.
