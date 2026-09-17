# Line 1: five-game Atari sequence, reversed order

Reversed order **SpaceInvaders → Boxing → Breakout → Pong → Q\*bert**. Seed 0,
greedy-100 evaluation, plain `impala_ac_multihead` net, identical task set,
thresholds, PPO hyperparameters and frame-matched budget for both methods.
Single seed, so no error bars.

Order sensitivity is reported in the paper as an observed phenomenon, with the
two orders' retention matrices as the evidence. Its root cause and any mitigation
are out of scope here, so the canonical order is **not** rebuilt in this folder;
`reports/order_sensitivity/` keeps that material.

## Status, 2026-09-17

Three of the four runs are complete. **Ours is the only one still going.**

| Method | Tasks finished | Source |
|---|---:|---|
| Min-Max (ours) | 4 / 5 | `data_live.json`, post-threshold-fix |
| CLEAR | 5 / 5 | `../../order_sensitivity/data.json` |
| CKA-RL | 5 / 5 | `data_live.json`, post-threshold-fix |
| CompoNet | 5 / 5 | `data_live.json`, post-threshold-fix |

Caveats that have to travel with any of these figures.

**Ours is measured after four tasks, the others after five.** Ours has not yet
taken the final Q\*bert consolidation, so its numbers are not strictly
comparable to the three finished runs and the difference runs in ours' favour.
Every figure says so, and nothing is extrapolated to close the gap.

**CLEAR's run predates the threshold change**, which does not affect it. CLEAR
never reached its thresholds on the first four games, so it trained to its full
budget on them regardless. Only Q\*bert cleared its threshold, and Q\*bert is
the last task, so nothing earlier is affected. CLEAR does not need rerunning.

**The threshold fix landed hard.** Ours' SpaceInvaders went 588.5 → 1318.1 on
task 1, which is 146% of the joint ceiling. That is the single largest change
in this tier and it is why the old and new runs are never averaged together.

**Ours' Breakout shows the value-vs-score gap.** The local phase peaked at 396,
the value-constraint shortfall reached ≈0, and the greedy diagonal still fell to
77.6 during consolidation. The constraint is satisfied on `V` while the score
collapses. It then recovered to 144.5 after the Pong phase, which is the only
positive backward transfer any method posts in this tier (+0.23). Worth stating
in the paper rather than leaving for a reviewer.

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

Ours sits between the two, retaining without freezing.

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

## How the unfinished run is shown

**One rule now, everywhere. Nothing is stood in for.**

Earlier versions padded a run's unreached tasks at ours' value, which was the
right call while ours was the most advanced run and the baselines were the ones
still going. That situation has reversed. Ours is now the only run in flight, so
the same rule would have put a **different run's** numbers under ours' own name
in the headline figure: ours' only complete five-task run is the
pre-threshold-fix one, whose task 1 scored 588.5 against the live run's 1318.1.

So an unreached task is simply absent. In `forgetting_matrices` ours' last row
is blank. In `final_scores` ours' Q\*bert slot is empty, with the slot itself
kept so the bars stay aligned across panels. `transfer_table` carries no
per-task rows at all, and every aggregate in it is computed over the tasks that
run has finished, with the count printed under each column. The columns are therefore not comparable
to each other, which the footnote says.

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
