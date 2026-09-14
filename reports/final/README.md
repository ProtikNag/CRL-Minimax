# Finalised paper figures

Curated, paper-ready figures only. Exploratory and diagnostic figure sets stay
where they were built (`reports/order_sensitivity/`, `reports/v5_clear_joint/`,
`diagnostics/`); nothing in those folders is a deliverable.

Two lines of comparison, one subfolder each.

| Subfolder | Line | Status |
|-----------|------|--------|
| `atari_reversed/` | Five-game Atari sequence, reversed order, ours vs CLEAR vs the Joint ceiling | done |
| `cka_rl/` | CKA-RL (NeurIPS 25) benchmark: Meta-World CW20, SpaceInvaders, Freeway | Table 1 + per-mode success done; Meta-World still running |

## Conventions shared by both lines

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
- **Single seed everywhere, so no error bars are drawn and none are invented.**

## Rebuild

```bash
python reports/final/atari_reversed/make_figures.py
```

Each subfolder's script is self-contained and reads cached numbers, so the
figures rebuild anywhere without the cluster run directories.
