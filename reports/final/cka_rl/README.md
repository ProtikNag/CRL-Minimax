# Line 2: CKA-RL (NeurIPS 2025) benchmark comparison

Our min-max method added as a row in the benchmark of Hu et al., *Continual
Knowledge Adaptation for Reinforcement Learning*, NeurIPS 2025
(`docs/papers/2025_Hu_CKA_RL_Continual_Knowledge_Adaptation.pdf`). The port lives
in the separate clone `CKA-RL-compare`, branch `ours-minmax-row`, not in this
repository.

## Figures

| Stem | What it shows | Status |
|------|---------------|--------|
| `table1_perf_fwt` | The paper's Table 1 (PERF and FWT across Meta-World, SpaceInvaders, Freeway) with our row added, sorted and highlighted | done |
| `final_policy_space_invaders` | Per-mode final return, spread, threshold and pass/fail, 10 modes | done |
| `final_policy_freeway` | Same, 8 modes | done |

All in `png/` (300 dpi) and `svg/` (vector).

## Table 1

Paper rows are transcribed in `data.json` from Table 1 on page 7. The
transcription is verified against the paper's own prose, which quotes eight
derived percentages; all eight reconcile (Average PERF 0.7196 → 0.7498 = +4.20%,
Average FWT 0.4674 → 0.5049 = +8.02%, Meta-World 0.4368 → 0.4642 = +6.27%,
SpaceInvaders PERF +1.02%, Freeway PERF +1.12%, SpaceInvaders FWT +11.29%,
Freeway FWT +1.73%, Meta-World FWT +41.82%).

Three decisions worth knowing about.

**The Average column is recomputed over SpaceInvaders and Freeway for every
row**, and is not the paper's three-environment Average. Our Meta-World run is
still going, and Meta-World is the low-scoring environment for every method
(0.00 to 0.46), so putting a two-environment mean for ours beside a
three-environment mean for everyone else would flatter ours by construction.
Recomputing the column the same way for all rows is the only way ours can be
ranked honestly. The paper's own Average is preserved in `data.json` under
`average_paper_3env`:

| Method | Paper Avg PERF (3 env) | Paper Avg FWT (3 env) |
|---|---:|---:|
| CKA-RL | 0.7498 | 0.5049 |
| CompoNet | 0.7196 | 0.4674 |
| FT-N | 0.7030 | 0.3886 |
| CReLUs | 0.6832 | 0.4174 |
| CbpNet | 0.6813 | 0.3740 |
| Baseline | 0.3917 | 0.0000 |
| ProgNet | 0.3680 | 0.0495 |
| PackNet | 0.2530 | −0.1834 |
| FT-1 | 0.2079 | 0.3886 |
| MaskNet | 0.1302 | −0.2688 |

No standard deviation is carried into the recomputed column. Combining two
per-environment standard deviations needs their seed-level covariance, which the
paper does not report, so any number there would be invented.

**Baseline is excluded from the forward-transfer bold.** Forward transfer is
defined against Baseline's own learning curve (paper eq. 9,
`FT_i = (AUC_i − AUC_i^b) / (1 − AUC_i^b)`), so Baseline scores exactly 0.0000 by
construction. It is the reference, not a competitor. The paper bolds it nowhere
in FWT either.

**Ours is quoted to three decimals**, which is the precision the run reports.
Padding to four would imply precision we do not have.

### Where ours lands

Sorted by the recomputed two-environment average performance:

| Rank | Method | Avg PERF | Avg FWT |
|---:|---|---:|---:|
| 1 | CKA-RL | 0.8925 | 0.7589 |
| 2 | CompoNet | 0.8729 | 0.7039 |
| **3** | **Min-Max (ours)** | **0.8670** | **0.6555** |
| 4 | FT-N | 0.8659 | 0.6900 |
| 5 | CReLUs | 0.8354 | 0.6305 |
| 6 | CbpNet | 0.8035 | 0.6022 |
| 7 | Baseline | 0.3780 | 0.0000 |
| 8 | ProgNet | 0.3441 | 0.0931 |
| 9 | FT-1 | 0.2962 | 0.6900 |
| 10 | PackNet | 0.2533 | 0.0610 |
| 11 | MaskNet | 0.0322 | −0.2185 |

Third on performance, sixth on forward transfer. Ours wins no individual column.

### Disclosures that belong in any caption

- **Ours has live past-task environment access**; no baseline does. It
  re-simulates and re-consolidates old modes, which is a stronger assumption
  than a replay buffer, not a weaker one.
- **Over 2× the frames per task.**
- **Seed 0 only**, against 10 seeds for every paper row.
- **Table 1 measures plasticity.** Retention is what our method is for, and it
  shows in the paper's Table 3, not here.
- Meta-World, when it lands, ran at a **reduced Δ of 300k steps** rather than the
  benchmark's 1M.

## Per-mode success detail

The paper's Table 3 reports one number per environment, the mean success rate.
These two figures show everything behind it: every mode's mean greedy return,
the episode-to-episode spread, the fixed threshold it had to clear, and the
resulting pass or fail.

Source: `raw/<env>/table3_final_policy.json`, copied from the comparison clone.
One figure per environment because 10 modes and 8 modes do not share a return
scale (SpaceInvaders runs to ~800, Freeway to ~26).

**Each row is a dumbbell, not a bar from zero.** The bar runs from that mode's
threshold to the achieved mean, so its length and direction *are* the margin,
and the dot marks the achieved return. Every threshold here is a different
number, so a bar from zero would spend most of its ink below the threshold where
nothing is being compared. Dropping the zero baseline also lets the axis cover
only the range the data occupies, which is what makes the near misses legible.
The axis therefore does not start at zero, and the figure says so.

| Environment | Cleared | Success rate | Mean return |
|---|---|---:|---:|
| SpaceInvaders | 10 of 10 | 1.00 | 592.1 |
| Freeway | 6 of 8 | 0.75 | 19.15 |

Freeway's two misses are modes 1 (16.8 against 19.2, ×0.88) and 4 (22.0 against
23.5, ×0.94). Both are near misses in score terms, and neither is close in
statistical terms: see below.

**The pale line is ±1 standard deviation across the 100 evaluation episodes**,
which is the spread of play, not the uncertainty on the mean. It sits under the
margin bar, so the part that shows is the overhang past the achieved mean, which
is the informative half. At n=100 the standard
error is a tenth of that. Every pass and fail here clears its threshold by at
least 5 standard errors, so no call in either figure is marginal:

| | closest call | margin |
|---|---|---|
| SpaceInvaders | mode 1, ×1.30 | far outside any interval |
| Freeway | mode 0, ×1.05 | 5.1 standard errors above |
| Freeway | mode 4, ×0.94 | 8.7 standard errors below |
| Freeway | mode 1, ×0.88 | 9.6 standard errors below |

Modes are drawn in learning order, which for both environments is the same as
mode index. Seed 0.

## Rebuild

```bash
python reports/final/cka_rl/make_figures.py
```
