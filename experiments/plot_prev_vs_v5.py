#!/usr/bin/env python3
"""Two-panel RECORD figure: previous min-max forgetting matrix vs updated V5.

RECORD FIGURE. Deliberately NOT apples-to-apples: the two panels use different
normalizations, different task orders, and are different runs. Each panel renders
exactly the numbers/sequence from its own authoritative source. The mixed-
normalization hint lives only in the output filename ("mixed_norm").

Data source (authoritative):
    results/combined_prev_vs_v5_figuredata.json

  LEFT  : threshold-normalized (random=0, threshold=1), RdYlGn, linear norm
          vmin=-0.1 vmax=1.1. Matrix values are read directly from the committed
          source figure (raw data not available on this branch) and rendered
          verbatim -- no recompute, no rescale.
  RIGHT : local-normalized (random=0, local=1), RdBu with TwoSlopeNorm centered
          at 0, vmax = max |value| over finite cells (matches the existing
          reports/atari5_v5 fig2_local_normalized styling).

Both panels are lower-triangular forgetting matrices: row = "after task k",
col = evaluated game. The upper triangle (games not yet seen) is masked (grey
for left / left unfilled+masked for right), never shown as 0. Every lower-tri
cell is annotated with its exact value (2 decimals); diagonal is bold.

Output:
    reports/atari5_v5/png/prev_minmax_vs_v5_fig2__mixed_norm.png
    reports/atari5_v5/svg/prev_minmax_vs_v5_fig2__mixed_norm.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "combined_prev_vs_v5_figuredata.json"
OUT_PNG = ROOT / "reports" / "atari5_v5" / "png"
OUT_SVG = ROOT / "reports" / "atari5_v5" / "svg"
BASENAME = "prev_minmax_vs_v5_fig2__mixed_norm"

MASK_GREY = "#dddddd"


def to_full_matrix(tri_rows, n):
    """Turn a list of lower-triangular rows into an (n, n) array with NaN in the
    upper triangle (not-yet-seen tasks)."""
    M = np.full((n, n), np.nan, dtype=float)
    for i, row in enumerate(tri_rows):
        for j, v in enumerate(row):
            M[i, j] = v
    return M


def _panel_common(ax, games, title, subtitle):
    n = len(games)
    ax.set_xticks(range(n))
    ax.set_xticklabels(games, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"after {g}" for g in games], fontsize=9)
    ax.set_xlabel("evaluated game", fontsize=10)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    # thin cell borders for legibility
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)


def annotate_cells(ax, M):
    n = M.shape[0]
    for i in range(n):
        for j in range(i + 1):  # lower triangle incl diagonal
            v = M[i, j]
            if not np.isfinite(v):
                continue
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight=weight, color="black",
            )


def draw_left(ax, spec):
    games = spec["games"]
    n = len(games)
    M = to_full_matrix(spec["matrix"], n)
    vmin, vmax = spec["vmin"], spec["vmax"]
    cmap = plt.get_cmap(spec["cmap"]).copy()
    cmap.set_bad(MASK_GREY)  # masked (not-yet-seen) upper triangle -> grey
    norm = Normalize(vmin=vmin, vmax=vmax)
    disp = np.ma.masked_invalid(M)
    im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="equal")
    _panel_common(ax, games, spec["title"], spec["subtitle"])
    annotate_cells(ax, M)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("threshold-normalized", fontsize=9)
    return M


def draw_right(ax, spec):
    games = spec["games"]
    n = len(games)
    M = to_full_matrix(spec["matrix"], n)
    finite = M[np.isfinite(M)]
    vmax = float(np.max(np.abs(finite)))  # max |value| over finite cells
    cmap = plt.get_cmap(spec["cmap"]).copy()
    cmap.set_bad(MASK_GREY)
    # TwoSlopeNorm centered at 0; symmetric so negatives render on the red side
    # and are never clipped (vmin = -vmax <= min).
    tsn = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    disp = np.ma.masked_invalid(M)
    im = ax.imshow(disp, cmap=cmap, norm=tsn, aspect="equal")
    _panel_common(ax, games, spec["title"], spec["subtitle"])
    annotate_cells(ax, M)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("local-normalized", fontsize=9)  # "local model", never "expert"
    return M, vmax


def main():
    data = json.loads(DATA.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    draw_left(axes[0], data["left"])
    draw_right(axes[1], data["right"])

    fig.suptitle(data["suptitle"], fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_PNG.mkdir(parents=True, exist_ok=True)
    OUT_SVG.mkdir(parents=True, exist_ok=True)
    png = OUT_PNG / f"{BASENAME}.png"
    svg = OUT_SVG / f"{BASENAME}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
