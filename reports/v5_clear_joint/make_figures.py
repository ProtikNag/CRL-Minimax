#!/usr/bin/env python3
"""
Publication-quality figure suite comparing three Atari continual-RL results.

Provenance: seed 0, greedy-100 eval, plain impala_ac_multihead net,
same 5 games/order (Qbert, Pong, Breakout, Boxing, SpaceInvaders),
max_ep_steps 10000. Single seed => NO cross-seed error bars.

Data (raw greedy-100 game scores, final after learning all 5 games):
  Local  : single-task specialist reference per game
  V5     : ours (min-max consolidation), last row of forgetting matrix
  CLEAR  : replay baseline, same net/budget
  Joint  : budget-matched 6M frames/game (30M total), multi-task ceiling

Faithfulness notes:
  - Boxing can be negative (range ~ -100..+100). V5 Boxing = -25.8 (forgot it,
    worse than random ~0). Negatives are shown as bars below 0, never clipped.
  - CLEAR Breakout = 0.0 is a real result (forgot Breakout).
  - No error bars are invented (single seed).
  - Retention ratios can be negative, zero, or >1; all shown honestly.

Transforms applied:
  - Figure 1: small multiples, one linear y-axis PER GAME (independent scales),
    because raw game scales differ ~200x (Pong ~20 vs Qbert ~4000). No log,
    no clipping, no normalization of the raw-score panels. Each panel's y-axis
    includes 0 so the sign of every bar is visible.
  - Figure 2: retention = method_score / reference_score (Local or Joint).
    This is a plain ratio (no smoothing/clipping). A dashed line at 1.0 marks
    "matches the reference". Ratios are annotated as percentages.
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
GAMES = ["Qbert", "Pong", "Breakout", "Boxing", "SpaceInvaders"]

SCORES = {
    "Local": np.array([4467.8, 20.0, 132.7, 94.0, 1132.2]),
    "V5":    np.array([4075.0, 19.8,  51.8, -25.8, 765.8]),
    "CLEAR": np.array([4350.0, 21.0,   0.0, 100.0, 800.0]),
    "Joint": np.array([4261.5, 20.7, 285.4, 67.5,  905.8]),
}

# Colorblind-safe palette (Wong 2011). Consistent per method across all figures.
COLORS = {
    "Local": "#000000",  # black  (specialist reference)
    "V5":    "#0072B2",  # blue   (ours)
    "CLEAR": "#E69F00",  # orange (replay baseline)
    "Joint": "#009E73",  # green  (multi-task ceiling)
}
LABELS = {
    "Local": "Local (specialist)",
    "V5":    "V5 (ours, min-max)",
    "CLEAR": "CLEAR (replay)",
    "Joint": "Joint (ceiling)",
}
METHOD_ORDER = ["Local", "V5", "CLEAR", "Joint"]

OUT = "/work/pnag/CRL-Minimax/reports/v5_clear_joint"
PNG = os.path.join(OUT, "png")
SVG = os.path.join(OUT, "svg")
for d in (PNG, SVG):
    os.makedirs(d, exist_ok=True)

CAPTION = ("seed 0, greedy-100, plain impala net, same games/order "
           "(Qbert, Pong, Breakout, Boxing, SpaceInvaders)")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(SVG, name + ".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- Actual per-game scores, small multiples (own y-axis per game)
# ---------------------------------------------------------------------------
def figure1():
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.6))
    x = np.arange(len(METHOD_ORDER))
    for gi, (ax, game) in enumerate(zip(axes, GAMES)):
        vals = [SCORES[m][gi] for m in METHOD_ORDER]
        cols = [COLORS[m] for m in METHOD_ORDER]
        bars = ax.bar(x, vals, color=cols, width=0.72,
                      edgecolor="black", linewidth=0.5)
        ax.set_title(game)
        ax.set_xticks(x)
        ax.set_xticklabels(["Local", "V5", "CLEAR", "Joint"],
                           rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)  # mark zero clearly

        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else abs(vmax) + 1
        lo = min(0, vmin) - 0.16 * rng
        hi = max(0, vmax) + 0.28 * rng  # extra headroom for value labels
        ax.set_ylim(lo, hi)

        # annotate each bar with its value; place negatives below the bar
        span = hi - lo
        for b, v in zip(bars, vals):
            offs = 0.02 * span
            if v >= 0:
                ax.text(b.get_x() + b.get_width() / 2, v + offs,
                        f"{v:g}", ha="center", va="bottom", fontsize=7.5,
                        rotation=90)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v - offs,
                        f"{v:g}", ha="center", va="top", fontsize=7.5,
                        rotation=90)
        if gi == 0:
            ax.set_ylabel("Raw greedy-100 score")

    legend_handles = [Patch(facecolor=COLORS[m], edgecolor="black", label=LABELS[m])
                      for m in METHOD_ORDER]
    fig.legend(handles=legend_handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Actual per-game scores (independent y-axis per game)",
                 y=1.15, fontsize=12)
    fig.text(0.5, -0.10, CAPTION, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    save(fig, "fig1_per_game_scores")


# ---------------------------------------------------------------------------
# Figure 2 -- Normalized retention, two panels
# ---------------------------------------------------------------------------
def _retention_panel(ax, ref_key, methods, title):
    x = np.arange(len(GAMES))
    n = len(methods)
    width = 0.8 / n
    ref = SCORES[ref_key]
    for mi, m in enumerate(methods):
        ratio = SCORES[m] / ref
        offset = (mi - (n - 1) / 2) * width
        bars = ax.bar(x + offset, ratio, width=width,
                      color=COLORS[m], edgecolor="black", linewidth=0.5,
                      label=f"{m} / {ref_key}")
        for b, r in zip(bars, ratio):
            va = "bottom" if r >= 0 else "top"
            off = 0.03 if r >= 0 else -0.03
            ax.text(b.get_x() + b.get_width() / 2, r + off,
                    f"{r*100:.0f}%", ha="center", va=va, fontsize=6.8,
                    rotation=90)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0,
               label=f"1.0 = matches {ref_key}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GAMES, rotation=25, ha="right")
    ax.set_ylabel(f"Retention (score / {ref_key})")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    _retention_panel(axes[0], "Local", ["V5", "CLEAR", "Joint"],
                     "(a) Retention vs LOCAL specialist")
    _retention_panel(axes[1], "Joint", ["V5", "CLEAR"],
                     "(b) Retention vs JOINT ceiling")
    # give headroom for the >1 bars (Joint Breakout/Local = 2.15) and negatives
    axes[0].set_ylim(-0.45, 2.45)
    axes[1].set_ylim(-0.45, 1.75)
    fig.suptitle("Normalized retention (single seed; ratios can be <0, 0, or >1)",
                 fontsize=12)
    fig.text(0.5, -0.06, CAPTION, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig2_retention")


# ---------------------------------------------------------------------------
# Figure 3 -- Compact mean-retention summary
# ---------------------------------------------------------------------------
def figure3():
    methods_local = ["V5", "CLEAR", "Joint"]
    methods_joint = ["V5", "CLEAR"]
    mean_local = {m: np.mean(SCORES[m] / SCORES["Local"]) for m in methods_local}
    mean_joint = {m: np.mean(SCORES[m] / SCORES["Joint"]) for m in methods_joint}

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    groups = ["vs Local", "vs Joint"]
    gx = np.arange(len(groups))
    all_methods = ["V5", "CLEAR", "Joint"]
    n = len(all_methods)
    width = 0.8 / n
    for mi, m in enumerate(all_methods):
        vals = [mean_local.get(m, np.nan), mean_joint.get(m, np.nan)]
        offset = (mi - (n - 1) / 2) * width
        xs = gx + offset
        bars = ax.bar(xs, vals, width=width, color=COLORS[m],
                      edgecolor="black", linewidth=0.5, label=m)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0,
               label="1.0 = matches reference")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(gx)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean retention across 5 games")
    ax.set_title("Mean retention summary")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.35)
    fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    save(fig, "fig3_mean_retention")

    return mean_local, mean_joint


if __name__ == "__main__":
    figure1()
    figure2()
    ml, mj = figure3()
    # Print a correspondence table for the self-audit.
    print("=== Raw scores ===")
    for m in METHOD_ORDER:
        print(f"{m:6s}", " ".join(f"{v:8.1f}" for v in SCORES[m]))
    print("\n=== Retention vs Local ===")
    for m in ["V5", "CLEAR", "Joint"]:
        r = SCORES[m] / SCORES["Local"]
        print(f"{m:6s}", " ".join(f"{v:7.3f}" for v in r))
    print("\n=== Retention vs Joint ===")
    for m in ["V5", "CLEAR"]:
        r = SCORES[m] / SCORES["Joint"]
        print(f"{m:6s}", " ".join(f"{v:7.3f}" for v in r))
    print("\n=== Mean retention ===")
    print("vs Local:", {k: round(v, 3) for k, v in ml.items()})
    print("vs Joint:", {k: round(v, 3) for k, v in mj.items()})
    print("\nWrote figures to", PNG, "and", SVG)
