# Presentation figures

Vector figures for talks. Every game screen is a real Arcade Learning Environment
frame, not a redrawing, converted losslessly into SVG rectangles.

| File | Content | Source |
|------|---------|--------|
| `pong_state.svg` / `.png` | Atari 2600 Pong, mid-rally, score 5-0 | `ALE/Pong-v5`, seed 7 |
| `breakout_state.svg` / `.png` | Atari 2600 Breakout, wall partly cleared, ball in play | `ALE/Breakout-v5`, seed 3 |
| `space_invaders_state.svg` / `.png` | Atari 2600 Space Invaders, formation descending | `ALE/SpaceInvaders-v5`, seed 11 |
| `robot.svg` / `.png` | Flat vector robot agent, academic palette | hand-authored |
| `gridworld_comparison.svg` / `.png` | Min-max consolidation vs fine-tuning on 20-task GridWorld | `reports/gridworld_20task/tables/`, 10 seeds |
| `atari_comparison.svg` / `.png` | Min-max consolidation vs CLEAR on 5 Atari games | v5 comparison figure, 2 seeds (see below) |

## The GridWorld figure

`make_gridworld_figure.py` reads the eval matrices and aggregated tables directly, so the
figure cannot drift from the results.

The left panel answers whether knowledge accumulates at all: how many already-learned
goals stay solvable after each new task is trained. Ours lands exactly on the
perfect-memory diagonal, fine-tuning flatlines near two, and the wedge between the two
curves is what the constraint saves. A goal counts as kept while its value stays within
10% of the value it had the moment its own task finished training, which is the matrix
diagonal. The right panel shows where each individual goal ended up after all 20 tasks,
with 95% CI bands over the 10 seeds.

Reported values, 10 seeds, greedy evaluation:

| Quantity | Ours (constrained min-max) | Fine-tuning |
| --- | --- | --- |
| Goals retained after task 20 | 20 / 20 | 2.9 / 20 |
| Mean success over 20 goals | 100.0% | 51.6% ± 4.0 |
| Mean steps to goal (150-step cap) | 11.3 | 87.9 |

The two retention counts use different definitions and both appear in the figure only
where labelled: the tile and left panel use value retention against the diagonal, while
the per-goal panel plots final greedy success rate.

## The Atari figure

`make_atari_figure.py` redraws the 5-task normalized forgetting matrices in the academic
palette (blue for ours, amber for CLEAR) and adds a drop chart: per game, a hollow marker
at the score it reached when it was trained, a solid marker at what survived the rest of
the sequence, so glyph length is the forgetting. The claim here is narrower than on
GridWorld and the figure says so — retention matches a replay method, absolute score does
not.

| Quantity | Ours (min-max) | CLEAR |
| --- | --- | --- |
| Forgetting (lower better) | 0.44 | 0.45 |
| Average performance | 0.45 | 0.73 |
| Stored frames | 0 | 8,192 |

Stored frames is `clear_snapshot_batches` (2) × `n_envs` (8) × `n_steps` (128) per task,
held for the four past tasks, from `crl/ppo/clear.py` and `configs/atari5_ppo_v5_clear.yaml`.

**Data provenance.** The full-budget v5 runs are not in this repo, so the two mean
matrices are transcribed from the generated figure at
`reports/atari5_ppo_v4/figures/clear_comparison/`. Recomputing the CL metrics from the
transcribed matrices reproduces the published values to within seed-averaging order
(AvgPerf 0.45 / 0.73, Forgetting 0.42 / 0.45, BWT −0.39 / −0.45), which is the check that
the transcription is faithful. If the v5 result directories are ever pulled back into
`results/`, replace the `MATRIX` constant with a loader over `eval_matrix.json`.

Regenerate with `.venv/bin/python presentation_figures/make_gridworld_figure.py`, then
rasterize as below.

## How the game figures were made

`render_atari_figures.py` rolls out a seeded policy, selects a frame that satisfies a
state predicate (ball mid-flight with both paddles inside the court for Pong, a visibly
chipped wall for Breakout), then vectorizes the 210x160 frame. ALE frames use a small
quantized palette, so horizontal run-length encoding followed by a vertical merge of
identical runs reproduces the frame exactly. Pong needs 16 rectangles, Breakout 33,
Space Invaders 540. Each SVG was verified pixel-identical to its source frame.

Breakout is driven by a ball-tracking heuristic rather than a random policy, since random
play never returns the ball and leaves the wall intact.

Scaling is unbounded. The SVGs carry `shape-rendering="crispEdges"`, so the hard pixel
edges stay hard at any projector size. The PNGs are 1280x1680 nearest-neighbour upscales
for slide tools that will not accept vector input.

The robot PNG is 2400x3040 with a transparent background, rasterized from `robot.svg`.

## Reproducing

Rendering needs `ale-py`, which is not in the project `.venv`. Either install it there or
keep it isolated:

```bash
.venv/bin/pip install --target /tmp/pylibs "ale-py>=0.10"
PYTHONPATH=/tmp/pylibs .venv/bin/python presentation_figures/render_atari_figures.py
```
