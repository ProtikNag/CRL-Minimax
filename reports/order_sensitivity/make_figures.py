#!/usr/bin/env python3
"""
Order-sensitivity figure suite: Local vs V5 vs Joint across two task orders
(canonical and reversed) for a 5-game continual-RL Atari result.

Provenance: seed 0, greedy-100 eval, plain impala_ac_multihead net, same 5
games. X-axis game order is fixed for comparability:
    Qbert, Pong, Breakout, Boxing, SpaceInvaders.

Series:
  Local : single-task specialist reference per game. ORDER-DEPENDENT
          (task-1 games have no local phase -> ref is the task-1 diagonal;
          later tasks' locals start from the evolving global). So
          Local-Canonical != Local-Reversed.
  V5    : min-max consolidation FINAL score on each game after learning all 5
          (last row of the forgetting matrix). Reported for both orders.
  Joint : one budget-matched model trained on all games at once ->
          ORDER-INDEPENDENT (same numbers in both orders). The fair, fixed
          cross-order reference.

Orders:
  Canonical = Qbert->Pong->Breakout->Boxing->SpaceInvaders
  Reversed  = SpaceInvaders->Boxing->Breakout->Pong->Qbert

Faithfulness notes / transforms:
  - Boxing can be NEGATIVE (V5-Canonical = -25.8: forgot Boxing, worse than
    random ~0). Negatives are drawn as bars below 0; a zero line is always
    drawn; y-limits always include 0. Never clipped.
  - Single seed => NO error bars are invented.
  - Fig 1: small multiples, INDEPENDENT linear y-axis per game (raw scales
    differ ~200x: Pong ~20 vs Qbert ~4000). No log, no clip, no normalization.
  - Figs 2/3: retention = V5_score / reference_score. Plain ratio, no
    smoothing/clipping. Dashed line at 1.0 = "matches reference". Ratios can be
    negative (Boxing), or >1 (retention above the reference); shown honestly.
  - Fig 2 denominator (Local) is order-dependent, so it mixes retention with
    local-reference differences. Fig 3 denominator (Joint) is fixed, so it is
    the cleaner cross-order comparison.
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data (seed 0, greedy-100, plain impala_ac_multihead net)
# ---------------------------------------------------------------------------
GAMES = ["Qbert", "Pong", "Breakout", "Boxing", "SpaceInvaders"]

# Per game, indexed by GAMES order above.
SCORES = {
    "Local_C": np.array([4467.8, 20.0, 132.7,  94.0, 1132.2]),
    "Local_R": np.array([4341.0, 21.0, 363.9,  98.9,  588.5]),
    "V5_C":    np.array([4075.0, 19.8,  51.8, -25.8,  765.8]),
    "V5_R":    np.array([4270.2, 21.0, 199.8,  55.4,  711.7]),
    "Joint":   np.array([4261.5, 20.7, 285.4,  67.5,  905.8]),
}

# Series order for the 5 bars per game in Fig 1.
FIG1_ORDER = ["Local_C", "Local_R", "V5_C", "V5_R", "Joint"]

# Colorblind-safe palette (Wong 2011). Color encodes the SERIES (Local/V5/Joint).
# Order (canonical vs reversed) is encoded by shade + hatch:
#   canonical = solid darker fill, no hatch
#   reversed  = lighter fill, "//" hatch
COL_LOCAL = "#000000"   # black  (specialist reference)
COL_V5    = "#0072B2"   # blue   (ours, min-max)
COL_JOINT = "#009E73"   # green  (multi-task ceiling, order-independent)
COL_LOCAL_R = "#7f7f7f"  # grey (lighter local for reversed)
COL_V5_R    = "#56B4E9"  # light blue for reversed V5

STYLE = {
    "Local_C": dict(color=COL_LOCAL,   hatch=None,  label="Local (canonical)"),
    "Local_R": dict(color=COL_LOCAL_R, hatch="//",  label="Local (reversed)"),
    "V5_C":    dict(color=COL_V5,      hatch=None,  label="V5 (canonical)"),
    "V5_R":    dict(color=COL_V5_R,    hatch="//",  label="V5 (reversed)"),
    "Joint":   dict(color=COL_JOINT,   hatch=None,  label="Joint (order-independent)"),
}

OUT = "/work/pnag/CRL-Minimax/reports/order_sensitivity"
PNG = os.path.join(OUT, "png")
SVG = os.path.join(OUT, "svg")
for d in (PNG, SVG):
    os.makedirs(d, exist_ok=True)

CAPTION = ("seed 0, greedy-100, plain impala net; "
           "Joint is order-independent, Local is order-dependent")

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
    "hatch.linewidth": 0.6,
})


def save(fig, name):
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(SVG, name + ".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- Actual per-game scores, small multiples (own y-axis per game)
# ---------------------------------------------------------------------------
def figure1():
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.8))
    x = np.arange(len(FIG1_ORDER))
    tick_labels = ["Local-C", "Local-R", "V5-C", "V5-R", "Joint"]
    for gi, (ax, game) in enumerate(zip(axes, GAMES)):
        vals = [SCORES[s][gi] for s in FIG1_ORDER]
        for bi, s in enumerate(FIG1_ORDER):
            st = STYLE[s]
            ax.bar(x[bi], vals[bi], width=0.72, color=st["color"],
                   hatch=st["hatch"], edgecolor="black", linewidth=0.5)
        ax.set_title(game)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)  # zero line always drawn

        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else abs(vmax) + 1
        lo = min(0, vmin) - 0.18 * rng   # always include 0
        hi = max(0, vmax) + 0.30 * rng   # headroom for rotated value labels
        ax.set_ylim(lo, hi)

        span = hi - lo
        for xi, v in zip(x, vals):
            offs = 0.02 * span
            if v >= 0:
                ax.text(xi, v + offs, f"{v:g}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90)
            else:
                ax.text(xi, v - offs, f"{v:g}", ha="center", va="top",
                        fontsize=7.5, rotation=90)
        if gi == 0:
            ax.set_ylabel("Raw greedy-100 score")

    legend_handles = [
        Patch(facecolor=STYLE[s]["color"], edgecolor="black",
              hatch=STYLE[s]["hatch"], label=STYLE[s]["label"])
        for s in FIG1_ORDER
    ]
    fig.legend(handles=legend_handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle("Figure 1 -- Actual per-game scores "
                 "(independent y-axis per game; each includes 0)",
                 y=1.17, fontsize=12)
    fig.text(0.5, -0.12, CAPTION, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    save(fig, "fig1_per_game_scores")


# ---------------------------------------------------------------------------
# Figures 2 & 3 -- Retention (V5 / reference), canonical vs reversed
# ---------------------------------------------------------------------------
def _retention_figure(ref_c, ref_r, ref_name, fig_no, fname, note):
    """Per game, two bars: V5-C/ref-C and V5-R/ref-R. Plus a MEAN group."""
    ratio_c = SCORES["V5_C"] / SCORES[ref_c]
    ratio_r = SCORES["V5_R"] / SCORES[ref_r]
    mean_c = float(np.mean(ratio_c))
    mean_r = float(np.mean(ratio_r))

    labels = GAMES + ["MEAN"]
    canon = list(ratio_c) + [mean_c]
    rev = list(ratio_r) + [mean_r]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars_c = ax.bar(x - width / 2, canon, width=width, color=COL_V5,
                    edgecolor="black", linewidth=0.5,
                    label=f"V5 canonical / {ref_name} (canonical)")
    bars_r = ax.bar(x + width / 2, rev, width=width, color=COL_V5_R,
                    hatch="//", edgecolor="black", linewidth=0.5,
                    label=f"V5 reversed / {ref_name} (reversed)")

    # Distinguish the MEAN group with a subtle boundary.
    ax.axvline(len(GAMES) - 0.5, color="0.7", linestyle=":", linewidth=1.0)

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.1,
               label=f"1.0 = matches {ref_name}")
    ax.axhline(0.0, color="black", linewidth=0.8)

    for bars, ratios in ((bars_c, canon), (bars_r, rev)):
        for b, r in zip(bars, ratios):
            va = "bottom" if r >= 0 else "top"
            off = 0.02 if r >= 0 else -0.02
            ax.text(b.get_x() + b.get_width() / 2, r + off,
                    f"{r*100:.0f}%", ha="center", va=va, fontsize=7.5,
                    rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"Retention  (V5 score / {ref_name})")
    ax.set_title(f"Figure {fig_no} -- Retention vs {ref_name.upper()} "
                 "(canonical vs reversed)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    lo = min(0.0, min(canon), min(rev))
    hi = max(1.0, max(canon), max(rev))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad + 0.12)

    fig.text(0.5, -0.14, CAPTION + "\n" + note, ha="center", fontsize=8,
             style="italic")
    fig.tight_layout()
    save(fig, fname)
    return ratio_c, ratio_r, mean_c, mean_r


def figure2():
    note = ("Denominator (Local) is ORDER-DEPENDENT, so this mixes retention "
            "with local-reference differences.")
    return _retention_figure("Local_C", "Local_R", "Local", 2,
                             "fig2_retention_vs_local", note)


def figure3():
    note = ("Denominator (Joint) is the FIXED, order-independent reference -> "
            "the cleaner cross-order comparison.")
    return _retention_figure("Joint", "Joint", "Joint", 3,
                             "fig3_retention_vs_joint", note)


if __name__ == "__main__":
    figure1()
    rc_l, rr_l, mc_l, mr_l = figure2()
    rc_j, rr_j, mc_j, mr_j = figure3()

    # Self-audit correspondence table.
    print("=== Raw scores (Qbert Pong Breakout Boxing SpaceInvaders) ===")
    for s in FIG1_ORDER:
        print(f"{s:8s}", " ".join(f"{v:8.1f}" for v in SCORES[s]))
    print("\n=== Retention vs Local (%) ===")
    print("canon ", " ".join(f"{v*100:6.0f}" for v in rc_l), f" | MEAN {mc_l*100:.0f}")
    print("rev   ", " ".join(f"{v*100:6.0f}" for v in rr_l), f" | MEAN {mr_l*100:.0f}")
    print("\n=== Retention vs Joint (%) ===")
    print("canon ", " ".join(f"{v*100:6.0f}" for v in rc_j), f" | MEAN {mc_j*100:.0f}")
    print("rev   ", " ".join(f"{v*100:6.0f}" for v in rr_j), f" | MEAN {mr_j*100:.0f}")
    print("\nWrote figures to", PNG, "and", SVG)
