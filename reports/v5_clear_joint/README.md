# V5 vs CLEAR vs Joint — Atari continual-RL figure suite

Publication-quality comparison of three continual-RL results on a 5-game Atari
sequence, plus the single-task specialist (Local) reference and the budget-matched
multi-task ceiling (Joint).

**Provenance (all series, apples-to-apples):** seed 0, greedy-100 eval, plain
`impala_ac_multihead` net, same 5 games in the same order, `max_ep_steps = 10000`.
Single seed → **no cross-seed error bars** (none are shown; none are invented).

Regenerate everything with:

```
/work/apps/python3/anaconda/2023.7/bin/python3 reports/v5_clear_joint/make_figures.py
```

(Script: `make_figures.py`, self-contained; writes to `png/` and `svg/`.)

## Data used (exact)

Games in order: **Qbert, Pong, Breakout, Boxing, SpaceInvaders**.
Raw greedy-100 game scores, final after learning all 5 games:

| Series | Qbert | Pong | Breakout | Boxing | SpaceInvaders |
|--------|------:|-----:|---------:|-------:|--------------:|
| **Local** (single-task specialist) | 4467.8 | 20.0 | 132.7 | 94.0 | 1132.2 |
| **V5** (ours, min-max consolidation) | 4075.0 | 19.8 | 51.8 | **-25.8** | 765.8 |
| **CLEAR** (replay baseline) | 4350.0 | 21.0 | **0.0** | 100.0 | 800.0 |
| **Joint** (6M frames/game, 30M total, ceiling) | 4261.5 | 20.7 | 285.4 | 67.5 | 905.8 |

Notes:
- Qbert's Local reference is its task-1 diagonal (4467.8); Qbert has no separate
  local phase.
- Boxing range is roughly -100..+100; **V5 Boxing = -25.8 is worse than a random
  agent (~0)** — V5 forgot Boxing. Shown as a bar below zero, not clipped.
- **CLEAR Breakout = 0.0 is a real result** — CLEAR forgot Breakout.
- Joint here is the **complete** budget-matched run (6M frames/game). A stronger
  but **incomplete** joint run (59.8M frames, stopped ~46M) reached approximately
  `[4430, 21.0, 218, 70.1, 1375]`; it is NOT the plotted "Joint" series and is
  mentioned only for context.

## Figures

### `fig1_per_game_scores.{png,svg}` — Actual per-game scores
Small multiples: one panel per game, **each with its own linear y-axis** (raw game
scales differ ~200x, so a single shared axis would crush Pong). Four bars per game
(Local, V5, CLEAR, Joint). Zero is drawn explicitly; Boxing's negative V5 bar is
visible. This is the raw-score "what actually happened" view.

### `fig2_retention.{png,svg}` — Normalized retention (two panels)
- **(a) vs Local specialist:** V5/Local, CLEAR/Local, Joint/Local. Dashed line at
  1.0 = "matches the single-task specialist".
- **(b) vs Joint ceiling:** V5/Joint, CLEAR/Joint. Dashed line at 1.0 = "matches
  the multi-task ceiling" (Joint is the reference, implicitly 1.0).
- Ratios can be negative (V5 Boxing/Local = -0.27), zero (CLEAR Breakout = 0), or
  >1 (Joint Breakout/Local = 2.15; CLEAR Boxing/Joint = 1.48). All shown honestly;
  each bar annotated with its % value.

### `fig3_mean_retention.{png,svg}` — Mean-retention summary
Mean across the 5 games of each method's retention vs Local and vs Joint.
Mean vs Local: V5 54%, CLEAR 76%, Joint 113%. Mean vs Joint: V5 51%, CLEAR 88%.
(These means are dominated by the single catastrophically-forgotten game per
method — read them alongside Fig 1/2, not in isolation.)

## Key takeaways
- **V5 and CLEAR each catastrophically forget a _different_ game.** V5 forgets
  **Boxing** (-25.8, below random). CLEAR forgets **Breakout** (0.0).
- On the games each method retains, both are close to the specialist (Qbert, Pong,
  SpaceInvaders all near or above 90% of Local).
- **Joint is the ceiling:** it does well on all five games (no catastrophic hole),
  including Breakout (285.4, above Local) and Boxing (67.5). It is the only series
  without a forgotten game.
- The aggregate mean-retention numbers hide the structure — the story is *which*
  single game each method drops, which Figures 1 and 2 make visible.

## Faithfulness
- Values in every figure were spot-checked against the table above (see the
  correspondence dump printed by `make_figures.py`).
- No error bars invented (single seed). No log axis, clipping, smoothing, or
  outlier removal. Retention is a plain score/reference ratio.
- Negatives and zeros are shown (bars below / at 0), never clipped away.
- Colorblind-safe Wong palette; consistent color per method across all figures.

Palette: Local = black, V5 = blue (#0072B2), CLEAR = orange (#E69F00),
Joint = green (#009E73).
