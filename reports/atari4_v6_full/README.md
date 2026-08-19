# atari4_v6_full — continual-learning result figures (seed 0)

## Reference model: the LOCAL model, NOT a stored expert

This is a **Part A ("experts NOT stored")** run. The per-task reference used for
all normalization is the run's **own LOCAL model** (`pi_L`) — the current-task
specialist / constraint target produced during that task's training — **not** an
external, separately-trained single-task expert. Earlier versions of these
figures wrongly normalized against stored experts; that is fixed here. The word
"expert" no longer appears as the reference anywhere in the figures or captions.

Data provenance: `results/atari4_v6_full_seed0/figure_data.json` (authoritative).
The forgetting-matrix scores are **greedy (argmax) evaluation over 100 rollouts**
(greedy-100). The per-task reference (LOCAL) scores come from `reference_scores`;
their meaning is given by `reference_label` (see the v1 provenance caveat below).

Regenerate everything with:

```bash
python experiments/make_figures.py \
    --run-dir results/atari4_v6_full_seed0 \
    --out reports/atari4_v6_full
```

Outputs are written to `png/` (300 DPI) and `svg/` (vector). The script depends
only on numpy + matplotlib, has no repo-internal imports, and is run-agnostic, so
a version-2 run can call it with a different `--run-dir`.

## Task arrival order

`Qbert -> Boxing -> Pong -> Breakout`

## Lower-triangular caveat (read this first)

The forgetting matrix is **lower-triangular**. Row `k` = the agent's state
*after* training task `k`; column `i` = its greedy-100 score on task `i`, for
`i <= k`. Future tasks are **never evaluated**, so the upper triangle is genuinely
absent — it is masked/hatched in every heatmap and must not be read as "zero" or
"not forgotten".

## v1 LOCAL-score provenance caveat (important)

The v1 run did **not** save the per-task local models. So `reference_scores` are
sourced heterogeneously:

- **Task 1 (Qbert):** `4217.0` is `global_after_task1`'s greedy-100 score — a
  genuine greedy-100 measurement of the just-trained model.
- **Tasks 2–4 (Boxing `36.3`, Pong `20.3`, Breakout `87.6`):** taken from the
  training logs as a **15-episode stop-eval**, not greedy-100. These are a
  coarser estimate of the local specialist's level and are **not** directly
  comparable in evaluation protocol to the greedy-100 forgetting-matrix scores.

Treat retention/normalization for tasks 2–4 as approximate for this reason.
Future runs that save local models should replace these with greedy-100 local
scores, at which point this caveat is removed.

## Normalization formulas

- **Local-normalized** (fig1 colors, fig2, fig4a, fig4b right):
  `(score - random) / (reference_local - random)`.
  `1.0` = the LOCAL specialist's level, `0.0` = random agent,
  `> 1.0` = **beats the local specialist** (positive backward transfer),
  `< 0` = **worse than random**.
- **Percentage-of-local** (fig3): `score / reference_local * 100`.

Per-game `reference_local` and `random` baselines come directly from
`figure_data.json` (`reference_scores`, `random_scores`).

## Honesty notes specific to this run

- **Qbert exceeds its local specialist (positive backward transfer).** Qbert's
  local reference is `4217.0` (task-1 greedy-100). Its greedy-100 global score
  *rises* across later tasks: `5353.0` after Pong and **`7148.0` after Breakout**,
  i.e. a ratio of `7148 / 4217 = 1.70` — a local-normalized value of **`1.72`**
  (well above `1.0`). This is rendered **honestly and un-clipped**: fig2 shows
  `1.28`/`1.72` (and `3.50` mid-sequence) as values above `1.0`, fig3 shows Qbert
  above the 100% line, and fig4b classifies Qbert's improvement as **backward
  transfer (negative/green bar)**, never as forgetting. The color scale extends
  past `1.0` rather than saturating at the local level.
- **Boxing collapses below random.** After Pong and after Breakout, Boxing scores
  are `-1.62` and `-3.61` (below the random baseline of `0.1`), i.e. worse than
  random → local-normalized `-0.05` and `-0.10`. Preserved literally: annotations
  show the true negative numbers, and the diverging colormap is centered at 0 so
  "below random" is visually distinct from "at random", never silently clipped.
  In fig4b this is the one true **forgetting** case (positive/orange bar).
- **Pong ~ local, Breakout partial.** Pong's final `19.75` vs local `20.3` is
  **~97%** of local (near-flat retention). Breakout's final `53.66` vs local
  `87.6` is **~61%** of local. Both are shown directly in fig3.

## Figure 5 (windowed agreement vs local model) — not present for v1

fig5 reads the **optional** `expert_agreement.json`. **No such file exists for the
v1 run, so fig5 is skipped.** When a future run supplies it, it is interpreted as
the **global-vs-LOCAL** `relative_gap` (lower = better; 0 = global matches/beats
the local model), captioned "windowed agreement vs local model", and reported
*alongside* the greedy-100 matrix, never alone.

## Figure index

| File | What it shows |
|------|---------------|
| `fig1_forgetting_matrix` | Raw greedy-100 scores, lower-triangular; cells annotated with raw score, colored by local-normalized value (diverging, centered at 0, extends above 1). |
| `fig2_local_normalized` | Local-normalized matrix `(score-random)/(local-random)`; 1=local, 0=random, >1 beats local, <0 worse than random. |
| `fig3_pct_local_retention` | Per-task `score/local*100` trajectory across the sequence; open circle = just-learned point. Pong ~97% flat, Boxing collapses below 0, Qbert >100% (beats local), Breakout ~61%. |
| `fig4a_avg_perf_over_tasks` | Mean local-normalized performance over seen tasks vs. number of tasks seen. |
| `fig4b_forgetting_bwt` | Forgetting per task = (just-learned - final), raw and local-normalized. Positive/orange = forgetting (Boxing), negative/green = backward transfer (Qbert). Last task excluded (no post-learning stage). |
| `fig5_local_agreement` | Windowed agreement vs local model (`relative_gap`, lower=better), lower-triangular. Only produced if `expert_agreement.json` exists — absent for v1. |

## Colormaps

- Diverging `RdBu` (centered at 0) for signed normalized matrices — colorblind-safe,
  0-anchored so "below random" is unambiguous; not saturated at 1 so ">local" is visible.
- `cividis_r` (perceptually uniform, colorblind-safe) for the agreement gap (fig5).
- Wong/Okabe-Ito categorical palette for line plots.
No rainbow/jet; no meaning encoded by hue alone (every cell is also annotated).
