#!/usr/bin/env python3
"""One-off comparison figure: V4 vs V5 continual-RL retention.

Reads results/v4_v5_retention_compare.json and regenerates a two-panel figure:
  (A) Grouped bar chart of FINAL-STATE retention (% of each run's OWN local
      specialist) per game, V4 vs V5, with 70% target and 100% parity lines.
  (B) The two full lower-triangular retention matrices as diverging heatmaps
      centered at 100% (specialist parity).

Retention % = 100 * global_score / local_ref, where local_ref is that run's
per-task greedy-100 LOCAL specialist score (NOT a stored expert). Each run is
normalized by ITS OWN local specialist (the intended retention-of-specialist
metric); V4 and V5 use different Boxing/SpaceInvaders specialists, so those two
games are not directly comparable in absolute-score terms across runs.

No clipping, smoothing, or transform is applied to the plotted values. Negative
retention (Boxing falls below random) renders honestly below zero.

Data provenance: results/v4_v5_retention_compare.json (greedy-100 eval scores).
Usage: python experiments/plot_v4_v5_compare.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "results", "v4_v5_retention_compare.json")
OUTDIR = os.path.join(ROOT, "reports", "v4_v5_compare")

# Colorblind-safe run colors (Okabe-Ito): blue (V4), orange (V5).
C_V4 = "#0072B2"
C_V5 = "#E69F00"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def load():
    with open(DATA) as f:
        d = json.load(f)
    games = d["games"]
    # Recompute retention from raw scores rather than trusting stored fields,
    # so the figure is derived directly from the authoritative eval matrix.
    out = {}
    for run in ("V4", "V5"):
        R = d[run]
        m = R["matrix"]
        lr = R["local_ref"]
        n = len(games)
        ret = np.full((n, n), np.nan)
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                ret[i, j] = 100.0 * v / lr[j]
        out[run] = {"ret": ret, "final": ret[-1].copy(), "local_ref": lr}
    return games, out


def panel_bars(ax, games, data):
    n = len(games)
    x = np.arange(n)
    w = 0.38
    v4 = data["V4"]["final"]
    v5 = data["V5"]["final"]

    b4 = ax.bar(x - w / 2, v4, w, color=C_V4, label="V4  (mu-cap only)",
                edgecolor="black", linewidth=0.5, zorder=3)
    b5 = ax.bar(x + w / 2, v5, w, color=C_V5, label="V5  (retention gate + mu-cap)",
                edgecolor="black", linewidth=0.5, zorder=3)

    # Reference lines: 70% retention target, 100% specialist parity, 0 baseline.
    ax.axhline(100, color="0.25", ls="--", lw=1.2, zorder=2)
    ax.axhline(70, color="#009E73", ls=":", lw=1.6, zorder=2)
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.text(-0.45, 100, "100% specialist parity", va="bottom", ha="left",
            fontsize=8.5, color="0.25")
    ax.text(-0.45, 70, "70% retention target", va="bottom", ha="left",
            fontsize=8.5, color="#009E73")

    # Value labels: above positive bars, below negative bars.
    for bars, vals in ((b4, v4), (b5, v5)):
        for rect, val in zip(bars, vals):
            xc = rect.get_x() + rect.get_width() / 2
            if val >= 0:
                ax.text(xc, val + 2.5, f"{val:.1f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
            else:
                ax.text(xc, val - 2.5, f"{val:.1f}", ha="center", va="top",
                        fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.set_ylabel("Final-state retention (% of run's own local specialist)")
    ax.set_title("(A) Final global model: retention per task", loc="left",
                 fontweight="bold")
    lo = min(v4.min(), v5.min())
    hi = max(v4.max(), v5.max())
    ax.set_ylim(lo - 18, hi + 16)
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)


def panel_heatmap(ax, games, ret, run_title, norm, cmap):
    n = len(games)
    masked = np.ma.masked_invalid(ret)  # upper triangle (not-yet-seen) hidden
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")
    cmap.set_bad("white")

    for i in range(n):
        for j in range(n):
            v = ret[i, j]
            if np.isnan(v):
                continue
            # Choose readable text color from the actual cell luminance so
            # deep-red (forgetting) and deep-blue (backward-transfer) cells
            # both get white text.
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            shade = "white" if lum < 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=8.5, color=shade)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(games, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels([f"after {g}" for g in games], fontsize=9)
    ax.set_xlabel("evaluated task")
    ax.set_title(run_title, loc="left", fontweight="bold", fontsize=11)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.85", lw=0.8)
    ax.tick_params(which="minor", length=0)
    return im


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    games, data = load()

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42,
                          wspace=0.28)

    # (A) spans both columns of the top row.
    axA = fig.add_subplot(gs[0, :])
    panel_bars(axA, games, data)

    # (B) two heatmaps, diverging colormap centered at 100% (parity).
    # RdBu_r: red = below parity (forgetting), white = parity, blue = above
    # (backward transfer). vmin below the lowest retention (-27.4) so negatives
    # render distinctly; vmax at 122 (max backward transfer).
    lo = min(data["V4"]["ret"][~np.isnan(data["V4"]["ret"])].min(),
             data["V5"]["ret"][~np.isnan(data["V5"]["ret"])].min())
    hi = max(data["V4"]["ret"][~np.isnan(data["V4"]["ret"])].max(),
             data["V5"]["ret"][~np.isnan(data["V5"]["ret"])].max())
    norm = TwoSlopeNorm(vmin=min(lo, -30), vcenter=100.0, vmax=max(hi, 122))
    # RdBu goes red(low)->white(mid)->blue(high). With vcenter=100, values
    # below parity map red (forgetting), above parity map blue (transfer).
    cmap = plt.get_cmap("RdBu")

    axB1 = fig.add_subplot(gs[1, 0])
    axB2 = fig.add_subplot(gs[1, 1])
    panel_heatmap(axB1, games, data["V4"]["ret"], "(B) V4 retention matrix",
                  norm, cmap)
    im = panel_heatmap(axB2, games, data["V5"]["ret"], "V5 retention matrix",
                       norm, cmap)

    cbar = fig.colorbar(im, ax=[axB1, axB2], fraction=0.035, pad=0.02,
                        ticks=[-30, 0, 50, 70, 100, 122])
    cbar.set_label("retention % (100 = local-specialist parity)")

    fig.suptitle(
        "V5 (retention gate + mu-cap) rescues the oldest task Qbert (15% -> 91%) "
        "vs V4 (mu-cap only);\nboth runs still fail Boxing (destroyed by the "
        "later SpaceInvaders consolidation) and Breakout",
        fontsize=13, fontweight="bold", y=0.985)

    fig.text(
        0.5, 0.005,
        "Retention = global score as % of that run's OWN greedy-100 local "
        "specialist (not an expert). Task order: Qbert->Pong->Breakout->Boxing"
        "->SpaceInv. Rows = global model after task k (lower-triangular; upper "
        "triangle = not-yet-seen, masked). V4 and V5 trained separate Boxing "
        "(68.5 vs 94.0) and SpaceInv (1062 vs 1132) specialists, so those two "
        "games' bars are not directly comparable across runs. Negatives are "
        "real and unclipped. Data: results/v4_v5_retention_compare.json.",
        ha="center", va="bottom", fontsize=7.6, color="0.3", wrap=True)

    png = os.path.join(OUTDIR, "v4_v5_retention_compare.png")
    svg = os.path.join(OUTDIR, "v4_v5_retention_compare.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    print("wrote", png)
    print("wrote", svg)


if __name__ == "__main__":
    main()
