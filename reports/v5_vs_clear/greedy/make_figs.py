#!/usr/bin/env python3
"""
Publication figures: V5 (min-max) vs CLEAR (3 buffer sizes) on a 5-task
sequential-Atari forgetting benchmark.

Data provenance:
    reports/v5_vs_clear/matrices_greedy.json
    eval = greedy-100 (as-run). Lower-triangular matrices; row k = model
    AFTER training task k, evaluated on tasks 0..k.

Transforms applied (disclosed):
    - Cell COLOR = per-task normalized retention = score / (that task's
      diagonal / just-learned value), clipped to [0, 1.2]. Raw score is
      annotated (never altered).
    - Where a task's diagonal (just-learned) value is 0 (Breakout under
      CLEAR A greedy eval), retention is UNDEFINED (0/0); such cells are
      drawn in a neutral hatched color, not fabricated as 1.0.

Faithfulness caveat encoded (verified, not speculation):
    greedy-100 argmax eval is unreliable for Breakout: for some policies the
    argmax never fires the ball -> spurious 0 for a competent policy. Every
    Breakout cell reading exactly 0 is marked with an asterisk. Numbers are
    NOT altered.

Outputs (PNG @200dpi + SVG) into this directory.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "matrices_greedy.json")

with open(DATA) as f:
    D = json.load(f)

GAMES = D["games"]
THRESH = D["thresholds"]
METHODS = [
    "V5 (ours, min-max)",
    "CLEAR A (buffer 41k)",
    "CLEAR B (buffer 164k)",
    "CLEAR C (buffer 492k)",
]
N = len(GAMES)

# ---- shared style ----
plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

# Colorblind-safe, perceptually-uniform sequential map (cividis).
CMAP = matplotlib.colormaps["cividis"]
RET_CLIP = 1.2
NORM = Normalize(vmin=0.0, vmax=RET_CLIP)
UNDEF_COLOR = "#7f7f7f"   # neutral grey for undefined retention (0/0)
FUTURE_COLOR = "#e8e8e8"  # upper triangle (future tasks, not evaluated)


def diag_values(mat):
    return [mat[k][k] for k in range(N)]


def is_breakout_zero(j, val):
    """Breakout column (index 2) cell that reads exactly 0 -> flag."""
    return (j == 2) and (val == 0.0)


def retention(score, diag):
    if diag == 0.0:
        return None  # undefined (0/0) -- do not fabricate
    return score / diag


def text_color_for(ret):
    """Pick readable annotation color given the cell's colormap value."""
    if ret is None:
        return "white"
    r, g, b, _ = CMAP(NORM(min(ret, RET_CLIP)))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.55 else "white"


CAVEAT = ("* Breakout greedy-100 is unreliable: argmax fails to fire the ball "
          "-> spurious 0 for a competent policy (e.g. CLEAR-A final "
          "greedy=0 vs stochastic=263). A stochastic-eval version follows.")


# ============================================================
# FIGURE 1: 2x2 forgetting matrices, one full lower-triangle per method
# ============================================================
def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0))
    axes = axes.ravel()

    row_labels = [f"after T{k+1}\n({GAMES[k]})" for k in range(N)]
    col_labels = [f"T{j+1}\n{GAMES[j]}" for j in range(N)]

    for ax, method in zip(axes, METHODS):
        mat = D["matrices"][method]
        diag = diag_values(mat)

        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.invert_yaxis()
        ax.set_aspect("equal")

        for k in range(N):          # row = after training task k
            for j in range(N):      # col = evaluated task j
                x0, y0 = j, k
                if j > k:
                    # future task -- not evaluated (upper triangle)
                    ax.add_patch(Rectangle((x0, y0), 1, 1,
                                           facecolor=FUTURE_COLOR,
                                           edgecolor="white", linewidth=1.5))
                    continue
                val = mat[k][j]
                ret = retention(val, diag[j])
                if ret is None:
                    face = UNDEF_COLOR
                    ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.5,
                                           hatch="///"))
                else:
                    face = CMAP(NORM(min(ret, RET_CLIP)))
                    ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=face,
                                           edgecolor="white", linewidth=1.5))

                # annotation = RAW score (unaltered)
                label = f"{val:.0f}"
                if is_breakout_zero(j, val):
                    label += "*"
                # mark the just-learned diagonal cell subtly
                weight = "bold" if j == k else "normal"
                ax.text(x0 + 0.5, y0 + 0.5, label, ha="center", va="center",
                        fontsize=10.5, color=text_color_for(ret),
                        fontweight=weight)

        # outline the diagonal (just-learned) cells
        for k in range(N):
            ax.add_patch(Rectangle((k, k), 1, 1, fill=False,
                                    edgecolor="#d62728", linewidth=2.2))

        ax.set_xticks(np.arange(N) + 0.5)
        ax.set_yticks(np.arange(N) + 0.5)
        ax.set_xticklabels(col_labels, fontsize=8.5)
        ax.set_yticklabels(row_labels, fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(method, fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xlabel("evaluated task", fontsize=9)
        ax.set_ylabel("model checkpoint", fontsize=9)

    # shared colorbar
    sm = cm.ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0, 1.2])
    cbar.set_label("retention = score / just-learned (diagonal) value  "
                   "[clipped 0-1.2]", fontsize=9)

    fig.suptitle("Sequential-Atari forgetting matrices  -  eval = greedy-100 "
                 "(as-run)\nCell text = raw score; cell colour = per-task "
                 "retention; red box = just-learned; grey = future task",
                 fontsize=12.5, y=0.985)

    legend = ("Red outline = diagonal (just-learned).  "
              "Grey hatched = retention undefined (just-learned score was 0).\n"
              + CAVEAT)
    fig.text(0.5, 0.015, legend, ha="center", va="bottom", fontsize=8.2,
             wrap=True)

    fig.subplots_adjust(left=0.11, right=0.90, top=0.90, bottom=0.10,
                        hspace=0.30, wspace=0.35)
    out_png = os.path.join(HERE, "fig1_forgetting_matrices.png")
    out_svg = os.path.join(HERE, "fig1_forgetting_matrices.svg")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_svg


# ============================================================
# FIGURE 2: grouped bars, final-row retention per task, ours vs CLEAR sweep
# ============================================================
def fig2():
    # colorblind-safe qualitative palette (Okabe-Ito subset)
    method_colors = {
        "V5 (ours, min-max)":   "#009E73",  # green (ours, highlighted)
        "CLEAR A (buffer 41k)": "#56B4E9",  # sky blue
        "CLEAR B (buffer 164k)": "#0072B2", # blue
        "CLEAR C (buffer 492k)": "#E69F00", # orange
    }

    # retention[method][task]; None if undefined (0/0)
    ret = {}
    undef = {}
    for m in METHODS:
        mat = D["matrices"][m]
        diag = diag_values(mat)
        final = mat[N - 1]
        rr, uu = [], []
        for j in range(N):
            r = retention(final[j], diag[j])
            if r is None:
                rr.append(0.0)
                uu.append(True)
            else:
                rr.append(r)
                uu.append(False)
        ret[m] = rr
        undef[m] = uu

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    x = np.arange(N)
    nb = len(METHODS)
    width = 0.20

    for i, m in enumerate(METHODS):
        offs = (i - (nb - 1) / 2) * width
        vals = ret[m]
        bars = ax.bar(x + offs, vals, width, label=m,
                      color=method_colors[m], edgecolor="black", linewidth=0.6,
                      zorder=3)
        for j, b in enumerate(bars):
            raw = D["matrices"][m][N - 1][j]
            # asterisk on Breakout spurious-zero cells
            star = "*" if is_breakout_zero(j, raw) else ""
            if undef[m][j]:
                # undefined retention (just-learned was 0) -> hatch + label
                b.set_hatch("///")
                b.set_facecolor("#cccccc")
                ax.text(b.get_x() + b.get_width() / 2, 0.02,
                        "n/a" + star, ha="center", va="bottom", fontsize=7.5,
                        rotation=90, color="black", zorder=4)
            else:
                lbl = f"{vals[j]:.2f}{star}"
                if vals[j] < 0:
                    # negative retention (e.g. V5 Boxing final = -25.76,
                    # diag 97). Bar shows true negative height; label below.
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() - 0.02, lbl, ha="center", va="top",
                            fontsize=7.5, zorder=4)
                else:
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.015, lbl, ha="center",
                            va="bottom", fontsize=7.5, zorder=4)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    ax.text(-0.45, 1.015, "full retention", ha="left", va="bottom",
            fontsize=8.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}" for g in GAMES])
    ax.set_ylabel("end-of-sequence retention\n(final score / just-learned score)")
    ax.set_xlabel("task")
    ax.set_ylim(-0.35, 1.3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), frameon=False)

    ax.set_title("End-of-sequence retention per task  -  V5 vs CLEAR buffer "
                 "sweep\neval = greedy-100 (as-run); retention clipped in "
                 "colour only, bars show true fraction", fontsize=11.5)

    note = ("Hatched 'n/a' = retention undefined (just-learned Breakout score "
            "was 0 under greedy eval).\n" + CAVEAT)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.2)

    fig.subplots_adjust(bottom=0.24, top=0.86)
    out_png = os.path.join(HERE, "fig2_final_retention.png")
    out_svg = os.path.join(HERE, "fig2_final_retention.svg")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_svg


if __name__ == "__main__":
    p1 = fig1()
    p2 = fig2()
    print("WROTE:")
    for p in (*p1, *p2):
        print(" ", p)
