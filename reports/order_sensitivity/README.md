# Order sensitivity: Local vs V5 vs Joint across two task orders

**Headline.** V5's forgetting is strongly **order-dependent**. In the canonical
order the Boxing result is a catastrophe (V5 retains **-27%** of Local, i.e. the
raw score is *negative*, -25.8), but under the **reversed** order the same game
is retained at **56%** (raw +55.4). Averaged over all 5 games, retention against
the **fixed joint ceiling** rises from **51% (canonical) to 86% (reversed)**.

## Provenance
seed 0, greedy-100 eval, plain `impala_ac_multihead` net, same 5 games.
Single seed => **no error bars are shown** (none are invented).

## Series
- **Local** — single-task specialist reference per game. **Order-dependent**:
  task-1 games have no local phase (their reference is the task-1 diagonal),
  and later tasks' locals start from the evolving global. So Local-Canonical
  != Local-Reversed (e.g. SpaceInvaders 1132.2 canonical vs 588.5 reversed).
- **V5** — min-max consolidation's FINAL score on each game after learning all
  5 (last row of the forgetting matrix). Reported for both orders.
- **Joint** — one budget-matched model trained on all games at once.
  **Order-independent** (identical in both orders): the fair, fixed cross-order
  reference / ceiling.

## Orders
- **Canonical** = Qbert -> Pong -> Breakout -> Boxing -> SpaceInvaders
- **Reversed**  = SpaceInvaders -> Boxing -> Breakout -> Pong -> Qbert

The x-axis is by **game** (same fixed order in every figure) for comparability,
independent of the learning order.

## Exact data (raw greedy-100 game scores)

| Game          | Local-C | Local-R | V5-C   | V5-R   | Joint  |
|---------------|--------:|--------:|-------:|-------:|-------:|
| Qbert         |  4467.8 |  4341.0 | 4075.0 | 4270.2 | 4261.5 |
| Pong          |    20.0 |    21.0 |   19.8 |   21.0 |   20.7 |
| Breakout      |   132.7 |   363.9 |   51.8 |  199.8 |  285.4 |
| Boxing        |    94.0 |    98.9 |  -25.8 |   55.4 |   67.5 |
| SpaceInvaders |  1132.2 |   588.5 |  765.8 |  711.7 |  905.8 |

Boxing V5-Canonical is **negative** (-25.8): V5 forgot Boxing worse than random
(~0) under the canonical order. This is shown honestly (bar below 0, zero line
drawn), never clipped.

## Retention (V5 / reference), %

**vs Local (order-dependent denominator):**

| Game          | canonical | reversed |
|---------------|----------:|---------:|
| Qbert         |       91% |      98% |
| Pong          |       99% |     100% |
| Breakout      |       39% |      55% |
| Boxing        |      -27% |      56% |
| SpaceInvaders |       68% |     121% |
| **MEAN**      |   **54%** |  **86%** |

**vs Joint (fixed, order-independent denominator — cleaner comparison):**

| Game          | canonical | reversed |
|---------------|----------:|---------:|
| Qbert         |       96% |     100% |
| Pong          |       96% |     101% |
| Breakout      |       18% |      70% |
| Boxing        |      -38% |      82% |
| SpaceInvaders |       85% |      79% |
| **MEAN**      |   **51%** |  **86%** |

(Ratios can be negative — Boxing canonical — or exceed 100% — e.g. SpaceInvaders
121% vs its reversed Local, or Pong 101% vs Joint. All shown as-is.)

## Figures
- **Figure 1 — Actual per-game scores** (`fig1_per_game_scores`). Small
  multiples: one panel per game, each with its **own** linear y-axis (raw scales
  differ ~200x: Pong ~20 vs Qbert ~4000). Each panel shows 5 bars
  (Local-C, Local-R, V5-C, V5-R, Joint) and **always includes 0**, so Boxing's
  negative V5-C bar is visible. No log / clip / normalization. Color = series
  (Local black/grey, V5 blue, Joint green); reversed order = lighter shade +
  `//` hatch.
- **Figure 2 — Retention vs LOCAL** (`fig2_retention_vs_local`). Per game two
  bars: V5-C/Local-C and V5-R/Local-R, plus a MEAN group; dashed line at 1.0.
  Caption warns the denominator (Local) is **order-dependent**, so this mixes
  retention with local-reference differences.
- **Figure 3 — Retention vs JOINT** (`fig3_retention_vs_joint`). Per game two
  bars: V5-C/Joint and V5-R/Joint, plus a MEAN group; dashed line at 1.0.
  Joint is the fixed, order-independent reference => the cleaner cross-order
  comparison. This is where the mean rises **51% -> 86%**.

## Files
- Script: `make_figures.py` (run with `/work/apps/python3/anaconda/2023.7/bin/python3`)
- PNG (200 dpi): `png/fig1_per_game_scores.png`, `png/fig2_retention_vs_local.png`, `png/fig3_retention_vs_joint.png`
- SVG (vector): `svg/fig1_per_game_scores.svg`, `svg/fig2_retention_vs_local.svg`, `svg/fig3_retention_vs_joint.svg`
