# atari4_v6_full — continual-learning result figures (seed 0)

Data provenance: `results/atari4_v6_full_seed0/figure_data.json` (authoritative)
and `results/atari4_v6_full_seed0/expert_agreement.json` (optional; drives fig5).
All scores are **greedy (argmax) evaluation over 100 rollouts**.

Regenerate everything with:

```bash
python experiments/make_figures.py \
    --run-dir results/atari4_v6_full_seed0 \
    --out reports/atari4_v6_full
```

Outputs are written to `png/` (300 DPI) and `svg/` (vector). The script depends
only on numpy + matplotlib and is run-agnostic, so a version-2 run can call it
with a different `--run-dir`.

## Task arrival order

`Qbert -> Boxing -> Pong -> Breakout`

## Lower-triangular caveat (read this first)

The forgetting matrix is **lower-triangular**. Row `k` = the agent's state
*after* training task `k`; column `i` = its greedy-100 score on task `i`, for
`i <= k`. Future tasks are **never evaluated**, so the upper triangle is genuinely
absent — it is masked/hatched in every heatmap and must not be read as "zero" or
"not forgotten".

## Normalization formulas

- **Expert-normalized** (fig1 colors, fig2, fig4a, fig4b right):
  `(score - random) / (expert - random)`.
  `1.0` = single-task expert, `0.0` = random agent, `< 0` = **worse than random**.
- **Percentage-of-expert** (fig3): `score / expert * 100`.

Per-game `expert` and `random` baselines come directly from
`figure_data.json` (`expert_scores`, `random_scores`).

## Honesty notes specific to this run

- **Boxing goes negative.** After Pong and after Breakout, Boxing scores are
  `-1.62` and `-3.61` (below the random baseline of `0.1`), i.e. worse than random.
  This is preserved literally: annotations show the true negative numbers, and
  the diverging colormap is centered at 0 so "below random" is visually distinct
  from "at random", never silently clipped.
- **Cross-game scale gap is not hidden.** Qbert's final score (7148) is only
  ~40% of its expert (17726.25); Breakout's final (53.66) is ~19% of its expert
  (285.4); Pong sits at ~99%. Expert normalization makes the four very different
  score scales comparable *without* flattening these gaps — fig2/fig3 show Qbert
  and Breakout clearly below expert while Pong is near 1.0.
- **Qbert exhibits backward transfer, not forgetting.** Its just-learned score
  (4217) is *lower* than its final score (7148); later tasks improved it. fig4b
  shows this as a negative bar (green), labeled as backward transfer, so it is not
  mistaken for forgetting.

## Expert-value discrepancy (fig5 only)

`figure_data.json` and `expert_agreement.json` were measured separately and
disagree on one baseline: Boxing expert = `65.74` (figure_data) vs `68.64`
(expert_agreement). Figures 1-4 use `figure_data.json` throughout. Figure 5 uses
`expert_agreement.json`'s own `relative_gap` values (which were computed against
that file's own expert scores), and is captioned as a separate windowed metric.
Report it *alongside* the greedy-100 matrix, never alone.

## Figure index

| File | What it shows |
|------|---------------|
| `fig1_forgetting_matrix` | Raw greedy-100 scores, lower-triangular; cells annotated with raw score, colored by expert-normalized value (diverging, centered at 0). |
| `fig2_expert_normalized` | Expert-normalized matrix `(score-random)/(expert-random)`; 1=expert, 0=random, <0 worse than random. |
| `fig3_pct_expert_retention` | Per-task `score/expert*100` trajectory across the sequence; open circle = just-learned point. Pong ~99% flat, Boxing collapses below 0, Qbert/Breakout partial. |
| `fig4a_avg_perf_over_tasks` | Mean expert-normalized performance over seen tasks vs. number of tasks seen. |
| `fig4b_forgetting_bwt` | Forgetting per task = (just-learned - final), raw and expert-normalized. Positive/orange = forgetting, negative/green = backward transfer. Last task excluded (no post-learning stage). |
| `fig5_expert_agreement` | Windowed expert-agreement matrix (`relative_gap`, lower=better), lower-triangular. Only produced if `expert_agreement.json` exists. |

## Colormaps

- Diverging `RdBu` (centered at 0) for signed normalized matrices — colorblind-safe,
  0-anchored so "below random" is unambiguous.
- `cividis_r` (perceptually uniform, colorblind-safe) for the agreement gap.
- Wong/Okabe-Ito categorical palette for line plots.
No rainbow/jet; no meaning encoded by hue alone (every cell is also annotated).
