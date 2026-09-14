# Line 1: five-game Atari sequence, reversed order

Reversed order **SpaceInvaders → Boxing → Breakout → Pong → Q\*bert**. Seed 0,
greedy-100 evaluation, plain `impala_ac_multihead` net, identical task set,
thresholds, PPO hyperparameters and frame-matched budget for both methods.
Single seed, so no error bars.

Order sensitivity is reported in the paper as an observed phenomenon, with the
two orders' retention matrices as the evidence. Its root cause and any mitigation
are out of scope here, so the canonical order is **not** rebuilt in this folder;
`reports/order_sensitivity/` keeps that material.

## Figures

| Stem | What it shows |
|------|---------------|
| `forgetting_matrices` | Full forgetting matrices, Min-Max against CLEAR, every cell as a fraction of the Joint ceiling |
| `final_scores` | Per-game final greedy-100 score, Local specialist and both methods as bars against the Joint ceiling drawn as a rule |
| `transfer_table` | Per-task backward transfer plus the aggregates, both methods |
| `compute_cost` | Wall-clock cost of one full five-game run, Min-Max against CLEAR and Joint |

All in `png/` (300 dpi) and `svg/` (vector).

## Transfer metrics

Scores go onto a common scale as `(raw - random) / (Joint ceiling - random)`
before any averaging. Raw backward transfer cannot be averaged across games whose
scores differ by roughly 700x. `random` is `RANDOM_SCORES` from
`crl/envs/atari.py`, read by the script so the one definition stays authoritative.

Backward transfer follows `analysis/continual_metrics.py`: for each task learned
before the last, `final - just_learned`. Negative is forgetting. The last task
has nothing trained after it, so it has no backward transfer and is excluded from
every mean.

| | Min-Max | CLEAR |
|---|---:|---:|
| Mean backward transfer | **−0.20** | −0.57 |
| Forgetting | **0.24** | 0.57 |
| Average performance, all 5 tasks | 0.85 | 1.11 |
| Average performance, prior 4 tasks | **0.82** | 0.47 |

Both average-performance rows are shown because the all-5 row inverts: CLEAR's
1.11 is its last-learned Q\*bert at 3.6x the ceiling carrying four forgotten
games. The prior-4 row is the one that measures retention.

Min-Max's Space Invaders backward transfer is **positive** (+0.16). It is the
first task in the sequence and it ends the run above where it was when it was
learned, so consolidation improved it rather than merely preserving it.

### Forward transfer is not measurable in this study

Reported as `—`, not estimated. Two independent reasons, either sufficient.

1. **No task is evaluated before it is trained.** `configs/atari5.yaml` sets
   `eval_all_tasks: false`, so every `eval_matrix.json` is strictly lower
   triangular ([ppo_continual.py:130](../../../crl/ppo_continual.py#L130)). The
   standard definition needs `R[i-1, i]`, which lives in the upper triangle.
2. **A zero-shot number would measure an untrained head, not transfer.**
   `ImpalaMultiHeadActorCriticPolicy` allocates every task's actor and critic
   head at construction
   ([impala.py:104](../../../crl/policies/impala.py#L104)), and a task's head
   stays at its random initialisation until that task arrives. Evaluating the
   task-`k-1` checkpoint on task `k` would therefore score a random head
   regardless of how much the shared trunk had transferred.

Flipping the flag and re-running does **not** fix this; reason 2 survives it.
Forward transfer in this architecture is a statement about *learning speed*, so
measuring it means the Continual World construction: the area between each task's
training curve under the continual learner and under a from-scratch single-task
learner. That needs per-task training curves for both methods plus from-scratch
references, none of which are in this repository for the reversed order.

The CKA-RL comparison (line 2) does report forward transfer, because that
benchmark ships a per-task train-from-scratch Baseline as the reference.

## Reading the retention colour

The scale is **clamped at the ceiling and pivoted at 65%**:

- below 65% of the ceiling renders as a tint of red, deepening toward 0%,
- 65% is the neutral point,
- above 65% warms toward blue, and **everything at or above 100% is the same
  blue**.

The clamp matters. The exploratory version ran the scale to 250% to fit CLEAR's
360% Q\*bert cell, which dragged genuinely retained cells such as 82% down into
pink. Colour is clamped; the **printed number is always the true value**,
however far above 100% it runs.

## Why the Joint ceiling is the denominator

**Joint** is one budget-matched model trained on all five games at once. It is
order-independent, so it is the same reference in both orders and keeps any
cross-order statement honest.

**Local** (the single-task specialist) is order-dependent: a task-1 game has no
local phase, and later locals start from the evolving global. SpaceInvaders is
1132.2 canonical against 588.5 reversed for the same game. Retention against
Local therefore mixes forgetting with reference drift, so Local appears here only
as a raw-score reference in `final_scores`, never as a denominator.

## What the two figures say

Min-Max holds the four prior tasks at 79 / 82 / 70 / 101% of the ceiling. CLEAR
holds the same four at 56 / 55 / 8 / 54% and loses Breakout outright, walking it
106% → 19% → 8% down the rows.

CLEAR's Q\*bert is the caveat: learned last, with nothing trained after it, it
reaches 15351 against a ceiling of 4262, which is 3.6×. That single cell lifts
CLEAR's five-game mean above Min-Max's while it is forgetting everything else,
which is why the **prior-task mean over the four earlier tasks (83% against 43%)
is the statistic to quote**, not the all-five mean (86% against 107%).

## Compute cost

Wall-clock hours for one complete five-game run, single GPU, seed 0. Values are
cached in `compute.json`, read off `report/figures/compute_cost_wall.svg`, since
the run directories are gitignored and live only on the cluster.

| | Hours | vs Min-Max |
|---|---:|---:|
| Joint | 13.2 | 0.73× |
| **Min-Max (ours)** | **18.1** | 1.00× |
| CLEAR | 36.9 | 2.04× |

CLEAR spans 36.4 to 37.3 across its replay-buffer configurations, drawn as a
capped span at the end of its stem; the configuration choice moves it by under
an hour.

Drawn as a lollipop with a dashed reference rule at Min-Max. With three values
the encoding is identical to a bar chart at a tenth of the ink, which leaves the
panel quiet enough to carry the rule, and the rule is what turns three numbers
into the comparison the reader came for. Zero is kept on the axis: unlike a
threshold comparison, hours have a meaningful zero and stem length is a real
magnitude.

Two things to carry into any caption.

**The two series are measured differently and the comparison is indicative, not
exact.** Ours sums per-phase `wall_s` from `resource_usage.json`, counting only
time inside the training phases. CLEAR and Joint take `t_wall` from the last row
of `logs.jsonl`, which is total elapsed and so also includes evaluation,
checkpointing and setup. Ours is the series understated by this, so the gap
shown is a lower bound on our advantage rather than an inflated one.

**`report/manifest.json` currently captions its own version of this figure as "V5
is substantially slower than CLEAR".** That contradicts the figure's own numbers
(15.1 h and 18.1 h against 36.4 to 37.3 h). The caption is stale and should be
corrected or dropped before the dashboard is shown to anyone.

## Data and rebuild

Numbers come from `reports/order_sensitivity/data.json`, the single transcription
of the cluster `eval_matrix.json` files. They are not duplicated here. Re-transcribe
that file if a run is rerun.

```bash
python reports/final/atari_reversed/make_figures.py
```
