# Line 1: five-game Atari sequence, reversed order

Reversed order **SpaceInvaders → Boxing → Breakout → Pong → Q\*bert**. Seed 0,
greedy-100 evaluation, plain `impala_ac_multihead` net, identical task set,
thresholds, PPO hyperparameters and frame-matched budget for both methods.
Single seed, so no error bars.

Order sensitivity is reported in the paper as an observed phenomenon, with the
two orders' retention matrices as the evidence. Its root cause and any mitigation
are out of scope here, so the canonical order is **not** rebuilt in this folder;
`reports/order_sensitivity/` keeps that material.

## Status, 2026-09-16

The reversed-order rerun is **in progress**, and this folder shows it mid-flight.

| Method | Tasks finished | Source |
|---|---:|---|
| Min-Max (ours) | 3 / 5 | `data_live.json`, post-threshold-fix |
| CLEAR | 5 / 5 | `../../order_sensitivity/data.json` |
| CKA-RL | 3 / 5 | `data_live.json`, post-threshold-fix |
| CompoNet | 2 / 5 | `data_live.json`, post-threshold-fix |

Three caveats that have to travel with any of these figures.

**Only CLEAR has finished.** Its run predates the threshold change, but that
does not affect it: CLEAR never reached its thresholds on the first four games,
so it trained to its full budget on them regardless. Only Q\*bert cleared its
threshold, and Q\*bert is the last task, so nothing earlier is affected. CLEAR
does not need rerunning.

**The threshold fix landed hard.** Ours' SpaceInvaders went 588.5 → 1318.1 on
task 1, which is 146% of the joint ceiling. That is the single largest change
in this tier and it is why the old and new runs are never averaged together.

**Ours' Breakout shows the value-vs-score gap.** The local phase peaked at 396,
the value-constraint shortfall reached ≈0, and the greedy diagonal still fell to
77.6 during consolidation. The constraint is satisfied on `V` while the score
collapses. Worth stating in the paper rather than leaving for a reviewer.

## Figures

| Stem | What it shows |
|---|---|
| `forgetting_matrices` | Retention matrices, 2×2, all four methods |
| `final_scores` | Per-game score after the final task, five series |
| `transfer_table` | Backward transfer and aggregates, four columns |

`backward_transfer_matrix` and `compute_cost` were dropped. Compute was measured
across different GPUs per method, so the comparison was never fair.

## How unfinished runs are shown

Two different rules, on purpose.

**The matrix and the bars pad.** A task a run has not reached is stood in for at
ours' value, as asked. In `final_scores` every bar is a solid fill in its series
colour, with no outline on any of them, and the stand-in is marked `‡` on the
value label. Mixing outlined and un-outlined bars in one panel read as two kinds
of thing before it read as measured against not. In `forgetting_matrices`, where
a cell has no label of its own to carry the mark, the stand-in is a faded fill
with a dotted border plus the `‡`.

**The table does not pad.** Its numbers are *derived*, and padding derived
numbers manufactures results: pairing ours' live diagonal with a padded final
row produced a backward transfer of −0.69 on SpaceInvaders for a run whose live
data shows SpaceInvaders **recovering** 498.5 → 793.2. So every number in the
table is computed over the tasks that run has actually finished, with the count
printed under each heading. The columns are therefore not comparable to each
other, which the footnote says.

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
