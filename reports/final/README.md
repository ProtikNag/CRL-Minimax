# Finalised paper figures

Curated, paper-ready figures only. Exploratory and diagnostic figure sets stay
where they were built (`reports/order_sensitivity/`, `reports/v5_clear_joint/`,
`diagnostics/`); nothing in those folders is a deliverable.

Three result sets. The two GridWorld/Atari tiers are the ones where tasks are
genuinely different and the head is shared; the CKA-RL tier is the published
benchmark, where tasks are modes of one game and each gets its own head.

| Subfolder | Line | Status |
|-----------|------|--------|
| `atari_reversed/` | Five-game Atari sequence, reversed order: ours vs CLEAR vs CKA-RL vs CompoNet | **in flight** — CLEAR, CKA-RL and CompoNet complete 5/5; ours at 4/5 |
| `gridworld/` | 50-task shared-head GridWorld: ours vs CKA-RL vs CbpNet vs CReLUs vs fine-tuning vs from-scratch | **done** — 6 methods x 3 seeds, 18 complete runs |
| `cka_rl/` | CKA-RL (NeurIPS 25) benchmark: Meta-World CW20, SpaceInvaders, Freeway | Table 1 + per-mode success done; Meta-World still running |

## Conventions shared by every set

- **The method is called Min-Max.** The internal "V5" version label does not
  appear in any finalised figure.
- **No titles inside the artwork.** The caption is set in LaTeX. A baked-in
  title duplicates it and is cropped at typesetting time anyway.
- **PNG and SVG for every figure**, in `png/` and `svg/`. PNG is 300 dpi at the
  intended print width; SVG is fully vector, with no embedded raster (which is
  why the matrix cells are drawn as shapes rather than as a Plotly heatmap,
  since Plotly rasterises heatmap traces).
- Academic template from `report/acviz.py`: white ground, Tufte spine,
  horizontal grid only, the fixed palette, Inter / JetBrains Mono.
- **Error bars are drawn only where seeds were measured.** GridWorld has three
  complete seeds for every method, so every interval there is measured. The
  Atari and CKA-RL sets are single-seed and carry none. No band is ever drawn
  over a single run.
- **Captions are derived, not typed.** Seed counts, tallies and quoted ranges
  are computed at render time, so a rebuild after new runs land updates the
  figures and their captions together.

## Rebuild

```bash
python reports/final/atari_reversed/make_figures.py
python reports/final/gridworld/make_figures.py
python reports/final/cka_rl/make_figures.py
```

Each subfolder's script is self-contained and reads cached numbers, so the
figures rebuild anywhere without the cluster run directories.
