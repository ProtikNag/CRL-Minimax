# Order sensitivity: Local vs V5 (min-max) vs CLEAR vs Joint across two task orders

**Headline.** V5's forgetting is strongly **order-dependent**. In the canonical
order the Boxing result is a catastrophe (V5 retains **-27%** of Local, i.e. the
raw score is *negative*, -25.8), but under the **reversed** order the same game
is retained at **56%** (raw +55.4). Averaged over all 5 games, V5 retention
against the **fixed joint ceiling** rises from **51% (canonical) to 86%
(reversed)**.

**CLEAR (baseline).** The frame-matched, apples-to-apples CLEAR baseline is *also*
order-sensitive but in a different pattern: it holds Boxing (100/100 canonical)
yet **collapses Breakout to 0** in canonical and heavily forgets it (23.9) in
reversed, and — because Qbert is the last task in the reversed order and CLEAR's
cloning constraint is weak — CLEAR lets **reversed Qbert run to 15350.8**, far
above every other model and reference (Joint ~4261). CLEAR's mean retention vs
Joint is 88% (canonical) and 107% (reversed, inflated by that Qbert outlier).

## Provenance
seed 0, greedy-100 eval, plain `impala_ac_multihead` net, same 5 games.
Single seed => **no error bars are shown** (none are invented).

CLEAR is **apples-to-apples** with V5: identical net, task set, thresholds, and
PPO hyperparameters; frame-matched budget; equal replay buffer
(`clear_snapshot_batches=4`, `clear_replay_task_per_step=8`). The only method
difference is CLEAR's replay + policy/value cloning vs V5's min-max dual
constraint.

## Series
- **Local** — single-task specialist reference per game. **Order-dependent**:
  task-1 games have no local phase (their reference is the task-1 diagonal),
  and later tasks' locals start from the evolving global. So Local-Canonical
  != Local-Reversed (e.g. SpaceInvaders 1132.2 canonical vs 588.5 reversed).
- **V5** — min-max consolidation's FINAL score on each game after learning all
  5 (last row of the forgetting matrix). Reported for both orders.
- **CLEAR** — the CLEAR baseline's FINAL score on each game after learning all 5
  (last row of its forgetting matrix). Reported for both orders.
- **Joint** — one budget-matched model trained on all games at once.
  **Order-independent** (identical in both orders): the fair, fixed cross-order
  reference / ceiling.

## Orders
- **Canonical** = Qbert -> Pong -> Breakout -> Boxing -> SpaceInvaders
- **Reversed**  = SpaceInvaders -> Boxing -> Breakout -> Pong -> Qbert

The x-axis is by **game** (same fixed order in every figure) for comparability,
independent of the learning order.

## Exact data (raw greedy-100 game scores)

| Game          | Local-C | Local-R | V5-C   | V5-R   | CLEAR-C | CLEAR-R  | Joint  |
|---------------|--------:|--------:|-------:|-------:|--------:|---------:|-------:|
| Qbert         |  4467.8 |  4341.0 | 4075.0 | 4270.2 |  4350.0 | 15350.8  | 4261.5 |
| Pong          |    20.0 |    21.0 |   19.8 |   21.0 |    21.0 |    11.1  |   20.7 |
| Breakout      |   132.7 |   363.9 |   51.8 |  199.8 |     0.0 |    23.9  |  285.4 |
| Boxing        |    94.0 |    98.9 |  -25.8 |   55.4 |   100.0 |    36.8  |   67.5 |
| SpaceInvaders |  1132.2 |   588.5 |  765.8 |  711.7 |   800.0 |   505.85 |  905.8 |

Boxing V5-Canonical is **negative** (-25.8): V5 forgot Boxing worse than random
(~0) under the canonical order. Shown honestly (bar below 0, zero line drawn),
never clipped. CLEAR-Reversed Qbert (15350.8) is a genuine **outlier** (last
task, uncapped); it is shown honestly and dominates the reversed-order y-scales.

CLEAR final scores are the last row of each order's `eval_matrix.json`:
- canonical `results/atari5_v5_clearA_equal_seed0` last row (cols Qbert, Pong,
  Breakout, Boxing, SpaceInv) = `[4350.0, 21.0, 0.0, 100.0, 800.0]`
- reversed `CRL-Minimax-joint/results/atari5_clear_order2_seed0` last row (cols
  SpaceInv, Boxing, Breakout, Pong, Qbert) = `[505.85, 36.8, 23.9, 11.1, 15350.75]`

## Retention (method / reference), %

**vs Local (order-dependent denominator):**

| Game          | V5-C | V5-R | CLEAR-C | CLEAR-R |
|---------------|-----:|-----:|--------:|--------:|
| Qbert         |  91% |  98% |     97% |    354% |
| Pong          |  99% | 100% |    105% |     53% |
| Breakout      |  39% |  55% |      0% |      7% |
| Boxing        | -27% |  56% |    106% |     37% |
| SpaceInvaders |  68% | 121% |     71% |     86% |
| **MEAN**      |  **54%** | **86%** | **76%** | **107%** |

**vs Joint (fixed, order-independent denominator — cleaner comparison):**

| Game          | V5-C | V5-R | CLEAR-C | CLEAR-R |
|---------------|-----:|-----:|--------:|--------:|
| Qbert         |  96% | 100% |    102% |    360% |
| Pong          |  96% | 101% |    101% |     54% |
| Breakout      |  18% |  70% |      0% |      8% |
| Boxing        | -38% |  82% |    148% |     55% |
| SpaceInvaders |  85% |  79% |     88% |     56% |
| **MEAN**      |  **51%** | **86%** | **88%** | **107%** |

(Ratios can be negative — Boxing V5-canonical — or exceed 100% — e.g.
SpaceInvaders V5 121% vs its reversed Local, Boxing CLEAR-C 148% vs Joint, or the
CLEAR-R Qbert ~360% last-task outlier. All shown as-is; CLEAR-R MEANs are
inflated by that Qbert cell.)

## Figures
- **Figure 1 — Actual per-game scores** (`fig1_per_game_scores`). Small
  multiples: one panel per game, each with its **own** linear y-axis (raw scales
  differ ~200x: Pong ~20 vs Qbert ~4000). Each panel shows **7 bars**
  (Local-C, Local-R, V5-C, V5-R, CLEAR-C, CLEAR-R, Joint) and **always includes
  0**, so Boxing's negative V5-C bar is visible. No log / clip / normalization.
  Color = series (Local black/grey, V5 blue, **CLEAR vermillion/orange**, Joint
  green); reversed order = lighter shade + `//` hatch.
- **Figure 2 — Retention vs LOCAL** (`fig2_retention_vs_local`). Per game **four
  bars** (V5-C, V5-R, CLEAR-C, CLEAR-R each / its own-order Local), plus a MEAN
  group; dashed line at 1.0. Caption warns the denominator (Local) is
  **order-dependent**, so this mixes retention with local-reference differences.
- **Figure 3 — Retention vs JOINT** (`fig3_retention_vs_joint`). Per game four
  bars (V5 and CLEAR, both orders, each / the fixed Joint), plus a MEAN group;
  dashed line at 1.0. Joint is the fixed, order-independent reference => the
  cleaner cross-order comparison. The CLEAR-R Qbert cell ~360% (last-task,
  uncapped) sets the y-scale and compresses the other bars (called out in the
  fig note).

## Retention matrices (forgetting matrix, task × training-phase)

Figures 4-6 are the full **forgetting matrices**, now a **2×2 grid**: rows =
method (**V5 top, CLEAR bottom**), columns = task order (**canonical left,
reversed right**). Within each panel: **rows** = training phase (the consolidated
model's score *after* learning task k); **columns** = the games in that run's
**learning order**; the **outlined diagonal** = the just-learned game right after
its consolidation; the **grey upper triangle** = a task not yet seen (never
evaluated, never imputed). Sources: the ragged `eval_matrix.json` of
`results/atari5_v5_seed0` (V5 canonical), `results/atari5_v5_order2_seed0`
(V5 reversed), `results/atari5_v5_clearA_equal_seed0` (CLEAR canonical), and
`CRL-Minimax-joint/results/atari5_clear_order2_seed0` (CLEAR reversed).

- **Figure 4 — RAW scores** (`fig4_retention_matrix_raw`). Cell text = raw
  greedy-100 score. Color = each cell as a **fraction of its own column's max**,
  computed **per panel** (a per-column visual aid, since raw game scales differ
  ~200x: Pong ~20 vs Qbert ~4000); no shared color scale, so read the numbers,
  not the shade.
- **Figure 5 — retention vs LOCAL** (`fig5_retention_matrix_vs_local`). Cell =
  score / that game's Local specialist. Diverging color centered at 1.0 (blue ≥
  reference, red < reference; white = 1.0), shared across **all four panels**.
- **Figure 6 — retention vs JOINT** (`fig6_retention_matrix_vs_joint`). Cell =
  score / the fixed order-independent Joint ceiling; shared diverging scale
  across all four panels — the cleaner comparison. Negative cells (Boxing V5
  canonical) render as the deepest red and are shown honestly, never clipped.

## Files
- Script: `make_figures.py`
- PNG (200 dpi): `png/fig1_per_game_scores.png`, `png/fig2_retention_vs_local.png`, `png/fig3_retention_vs_joint.png`,
  `png/fig4_retention_matrix_raw.png`, `png/fig5_retention_matrix_vs_local.png`, `png/fig6_retention_matrix_vs_joint.png`
- SVG (vector): `svg/fig1_per_game_scores.svg`, `svg/fig2_retention_vs_local.svg`, `svg/fig3_retention_vs_joint.svg`,
  `svg/fig4_retention_matrix_raw.svg`, `svg/fig5_retention_matrix_vs_local.svg`, `svg/fig6_retention_matrix_vs_joint.svg`
</content>
</invoke>
