#!/usr/bin/env python3
"""
Order-sensitivity figure suite: Local vs V5 (ours, min-max) vs CLEAR vs Joint
across two task orders (canonical and reversed) for a 5-game continual-RL Atari
result.

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
  CLEAR : the CLEAR baseline (replay + policy/value cloning), frame-matched and
          apples-to-apples with V5 (identical net, task set, thresholds, PPO
          hyperparameters; equal buffer snapshot_batches=4, replay=8). FINAL
          score on each game after learning all 5 (last row of its forgetting
          matrix). Reported for both orders.
          Source: results/atari5_v5_clearA_equal_seed0/eval_matrix.json
          (canonical) and CRL-Minimax-joint results/atari5_clear_order2_seed0/
          eval_matrix.json (reversed).
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
  - CLEAR-Reversed Qbert = 15350.8 is a genuine OUTLIER: Qbert is the LAST task
    in the reversed order, so CLEAR (weak cloning constraint) lets it run far
    above every other model/reference (Joint ~4261). It is shown honestly, not
    clipped; the retention-vs-Joint bar for that cell reaches ~360% and is
    labelled. This compresses the other bars -- called out in the fig note.
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
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import TwoSlopeNorm

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
    # CLEAR final scores = last row of each order's eval_matrix, re-indexed to
    # the fixed GAMES column order [Qbert, Pong, Breakout, Boxing, SpaceInv].
    #   canonical last row (cols Qbert,Pong,Breakout,Boxing,SpaceInv):
    #       [4350.0, 21.0, 0.0, 100.0, 800.0]
    #   reversed last row (cols SpaceInv,Boxing,Breakout,Pong,Qbert):
    #       [505.85, 36.8, 23.9, 11.1, 15350.75]  -> reindexed below
    "CLEAR_C": np.array([4350.0, 21.0,  0.0, 100.0,  800.0]),
    "CLEAR_R": np.array([15350.75, 11.1, 23.9, 36.8, 505.85]),
    "Joint":   np.array([4261.5, 20.7, 285.4,  67.5,  905.8]),
}

# Series order for the bars per game in Fig 1.
FIG1_ORDER = ["Local_C", "Local_R", "V5_C", "V5_R",
              "CLEAR_C", "CLEAR_R", "Joint"]

# ---------------------------------------------------------------------------
# Forgetting / retention matrices (Figs 4-6). V5 min-max, seed 0, greedy-100.
#
# Rows  = training phase (score of the consolidated model AFTER learning task k).
# Cols  = tasks in the LEARNING order of that run (diagonal = the just-learned
#         game right after its consolidation; upper triangle = not-yet-seen,
#         left blank -- eval_all_tasks=false, we never evaluate a future task).
# Source: results/atari5_v5_seed0/eval_matrix.json (canonical) and
#         results/atari5_v5_order2_seed0/eval_matrix.json (reversed).
# NaN marks the unobserved upper triangle; it is masked (grey), never imputed.
_NAN = np.nan

# Canonical learning order: Qbert -> Pong -> Breakout -> Boxing -> SpaceInvaders
GAMES_C = ["Qbert", "Pong", "Breakout", "Boxing", "SpaceInv"]
ROWS_C = ["after Qbert", "after Pong", "after Breakout",
          "after Boxing", "after SpaceInv"]
MAT_C = np.array([
    [4467.75,   _NAN,   _NAN,   _NAN,   _NAN],
    [4145.50,  19.75,   _NAN,   _NAN,   _NAN],
    [5448.00,  20.00,  90.45,   _NAN,   _NAN],
    [ 729.00,  18.96,  44.08,  97.00,   _NAN],
    [4075.00,  19.75,  51.81, -25.76, 765.85],
])
# References in the SAME (canonical) column order.
LOCAL_C = np.array([4467.8, 20.0, 132.7,  94.0, 1132.2])   # order-dependent
JOINT_C = np.array([4261.5, 20.7, 285.4,  67.5,  905.8])   # order-independent

# Reversed learning order: SpaceInvaders -> Boxing -> Breakout -> Pong -> Qbert
GAMES_R = ["SpaceInv", "Boxing", "Breakout", "Pong", "Qbert"]
ROWS_R = ["after SpaceInv", "after Boxing", "after Breakout",
          "after Pong", "after Qbert"]
MAT_R = np.array([
    [588.50,   _NAN,   _NAN,  _NAN,    _NAN],
    [587.50,  96.85,   _NAN,  _NAN,    _NAN],
    [595.50,  21.72, 293.58,  _NAN,    _NAN],
    [702.40,  74.19, 156.73, 21.00,    _NAN],
    [711.65,  55.44, 199.83, 21.00, 4270.25],
])
# References in the SAME (reversed) column order (Joint is the canonical Joint
# values re-indexed to this column order -- it is order-independent).
LOCAL_R = np.array([588.5, 98.9, 363.9, 21.0, 4341.0])
JOINT_R = np.array([905.8, 67.5, 285.4, 20.7, 4261.5])

# CLEAR forgetting matrices (same layout as MAT_C / MAT_R above): rows = phase,
# cols = tasks in that run's LEARNING order, diagonal outlined, upper triangle
# NaN (unobserved). Verbatim from each run's eval_matrix.json (greedy-100).
#   canonical: results/atari5_v5_clearA_equal_seed0/eval_matrix.json
#   reversed : CRL-Minimax-joint/results/atari5_clear_order2_seed0/eval_matrix.json
MAT_CLEAR_C = np.array([
    [4375.00,   _NAN,   _NAN,   _NAN,   _NAN],
    [4350.00,  21.00,   _NAN,   _NAN,   _NAN],
    [4296.00,  21.00,   0.00,   _NAN,   _NAN],
    [4306.00,  14.33,  30.85, 100.00,   _NAN],
    [4350.00,  21.00,   0.00, 100.00, 800.00],
])
MAT_CLEAR_R = np.array([
    [588.50,   _NAN,   _NAN,  _NAN,     _NAN],
    [532.60, 100.00,   _NAN,  _NAN,     _NAN],
    [484.80,  88.00, 303.63,  _NAN,     _NAN],
    [529.90,  69.94,  54.92, 21.00,     _NAN],
    [505.85,  36.80,  23.90, 11.10, 15350.75],
])

# Colorblind-safe palette (Wong 2011). Color encodes the SERIES (Local/V5/Joint).
# Order (canonical vs reversed) is encoded by shade + hatch:
#   canonical = solid darker fill, no hatch
#   reversed  = lighter fill, "//" hatch
COL_LOCAL = "#000000"   # black  (specialist reference)
COL_V5    = "#0072B2"   # blue   (ours, min-max)
COL_JOINT = "#009E73"   # green  (multi-task ceiling, order-independent)
COL_LOCAL_R = "#7f7f7f"  # grey (lighter local for reversed)
COL_V5_R    = "#56B4E9"  # light blue for reversed V5
COL_CLEAR   = "#D55E00"  # vermillion (CLEAR baseline, canonical)
COL_CLEAR_R = "#E69F00"  # orange     (CLEAR baseline, reversed)

STYLE = {
    "Local_C": dict(color=COL_LOCAL,   hatch=None,  label="Local (canonical)"),
    "Local_R": dict(color=COL_LOCAL_R, hatch="//",  label="Local (reversed)"),
    "V5_C":    dict(color=COL_V5,      hatch=None,  label="V5 (canonical)"),
    "V5_R":    dict(color=COL_V5_R,    hatch="//",  label="V5 (reversed)"),
    "CLEAR_C": dict(color=COL_CLEAR,   hatch=None,  label="CLEAR (canonical)"),
    "CLEAR_R": dict(color=COL_CLEAR_R, hatch="//",  label="CLEAR (reversed)"),
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
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.0))
    x = np.arange(len(FIG1_ORDER))
    tick_labels = ["Local-C", "Local-R", "V5-C", "V5-R",
                   "CLEAR-C", "CLEAR-R", "Joint"]
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
    fig.legend(handles=legend_handles, ncol=7, loc="upper center",
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
    """Per game, FOUR bars -- {V5,CLEAR} x {canonical,reversed} -- each divided
    by its own-order reference. Plus a MEAN group. Ratios shown honestly:
    negatives (Boxing V5-C) draw below 0; >1 (CLEAR-R Qbert ~360%) not clipped.
    """
    # (score_key, ref_key, style_key, label) -- one entry per drawn bar series.
    series = [
        ("V5_C",    ref_c, "V5_C",    f"V5 / {ref_name} (canonical)"),
        ("V5_R",    ref_r, "V5_R",    f"V5 / {ref_name} (reversed)"),
        ("CLEAR_C", ref_c, "CLEAR_C", f"CLEAR / {ref_name} (canonical)"),
        ("CLEAR_R", ref_r, "CLEAR_R", f"CLEAR / {ref_name} (reversed)"),
    ]
    labels = GAMES + ["MEAN"]
    x = np.arange(len(labels))
    nser = len(series)
    width = 0.80 / nser

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ratios_out = {}
    all_vals = []
    for si, (score_key, rkey, style_key, lab) in enumerate(series):
        ratio = SCORES[score_key] / SCORES[rkey]
        vals = list(ratio) + [float(np.mean(ratio))]
        ratios_out[score_key] = ratio
        st = STYLE[style_key]
        offset = (si - (nser - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, color=st["color"],
                      hatch=st["hatch"], edgecolor="black", linewidth=0.5,
                      label=lab)
        all_vals.extend(vals)
        for b, r in zip(bars, vals):
            va = "bottom" if r >= 0 else "top"
            off = 0.02 if r >= 0 else -0.02
            ax.text(b.get_x() + b.get_width() / 2, r + off,
                    f"{r*100:.0f}%", ha="center", va=va, fontsize=6.5,
                    rotation=90)

    # Distinguish the MEAN group with a subtle boundary.
    ax.axvline(len(GAMES) - 0.5, color="0.7", linestyle=":", linewidth=1.0)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.1,
               label=f"1.0 = matches {ref_name}")
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"Retention  (method score / {ref_name})")
    ax.set_title(f"Figure {fig_no} -- Retention vs {ref_name.upper()}: "
                 "V5 vs CLEAR (canonical vs reversed)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)

    lo = min(0.0, min(all_vals))
    hi = max(1.0, max(all_vals))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad + 0.12)

    fig.text(0.5, -0.16, CAPTION + "\n" + note, ha="center", fontsize=8,
             style="italic")
    fig.tight_layout()
    save(fig, fname)
    # Preserve the original return contract (V5 ratios + means) for the audit.
    rc, rr = ratios_out["V5_C"], ratios_out["V5_R"]
    return rc, rr, float(np.mean(rc)), float(np.mean(rr))


def figure2():
    note = ("Denominator (Local) is ORDER-DEPENDENT, so this mixes retention "
            "with local-reference differences.")
    return _retention_figure("Local_C", "Local_R", "Local", 2,
                             "fig2_retention_vs_local", note)


def figure3():
    note = ("Denominator (Joint) is the FIXED, order-independent reference -> "
            "the cleaner cross-order comparison. CLEAR-reversed Qbert ~360% "
            "(last-task, uncapped) sets the y-scale and compresses other bars.")
    return _retention_figure("Joint", "Joint", "Joint", 3,
                             "fig3_retention_vs_joint", note)


# ---------------------------------------------------------------------------
# Figures 4-6 -- Retention / forgetting MATRICES (task x training-phase).
#   Fig 4: raw greedy-100 scores (color = per-column fraction-of-max, since raw
#          scales differ ~200x across games; cell text = the RAW score).
#   Fig 5: normalized by LOCAL reference (retention).
#   Fig 6: normalized by JOINT reference (retention).
# Normalized cells use a diverging colormap centered at 1.0 (= matches the
# reference); a shared color scale spans BOTH orders so the two panels are
# directly comparable. Negative cells (Boxing canonical) are shown honestly.
# ---------------------------------------------------------------------------
def _lower_mask(M):
    """True where a cell is unobserved (upper triangle / NaN)."""
    return ~np.isfinite(M)


def _draw_matrix(ax, M, row_labels, col_labels, kind,
                 ref=None, norm=None, cmap=None):
    """Draw one heatmap. kind in {'raw','norm'}.

    raw : color = value / column-max (per-column, purely a visual aid because
          raw game scales differ ~200x); annotated with the RAW score.
    norm: color-array = value / ref (broadcast over columns); TwoSlopeNorm
          centered at 1.0; annotated with the retention percentage.
    Unobserved cells (upper triangle) are masked grey and left blank.
    """
    n = M.shape[0]
    mask = _lower_mask(M)
    if kind == "raw":
        with np.errstate(invalid="ignore"):
            colmax = np.nanmax(np.where(mask, np.nan, M), axis=0)
        C = M / colmax                      # per-column [.,1]
        cmap = cmap or plt.cm.viridis.copy()
        vmin, vmax = 0.0, 1.0
        cmap.set_bad("0.9")
        im = ax.imshow(np.ma.array(C, mask=mask), cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
    else:                                    # normalized retention
        C = M / ref                          # broadcast ref over columns
        cmap = cmap or plt.cm.RdBu.copy()
        cmap.set_bad("0.9")
        im = ax.imshow(np.ma.array(C, mask=mask), cmap=cmap,
                       norm=norm, aspect="auto")

    # Cell annotations (white bbox for legibility over any color).
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue
            if kind == "raw":
                txt = f"{M[i, j]:g}"
            else:
                txt = f"{C[i, j] * 100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              alpha=0.60, edgecolor="none"))

    # Outline the diagonal (the just-learned game before later interference).
    for d in range(n):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.6))

    ax.set_xticks(range(n))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("evaluated on task (learning order ->)")
    ax.tick_params(length=0)
    return im


def figure4_raw():
    """Raw forgetting matrices: rows = method (V5 / CLEAR), cols = order."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    _draw_matrix(axes[0, 0], MAT_C,       ROWS_C, GAMES_C, "raw")
    _draw_matrix(axes[0, 1], MAT_R,       ROWS_R, GAMES_R, "raw")
    _draw_matrix(axes[1, 0], MAT_CLEAR_C, ROWS_C, GAMES_C, "raw")
    _draw_matrix(axes[1, 1], MAT_CLEAR_R, ROWS_R, GAMES_R, "raw")
    axes[0, 0].set_title("V5 (min-max) -- Canonical order")
    axes[0, 1].set_title("V5 (min-max) -- Reversed order")
    axes[1, 0].set_title("CLEAR -- Canonical order")
    axes[1, 1].set_title("CLEAR -- Reversed order")
    axes[0, 0].set_ylabel("training phase")
    axes[1, 0].set_ylabel("training phase")
    fig.suptitle("Figure 4 -- Forgetting matrix: RAW greedy-100 score "
                 "(diagonal = just-learned game); V5 (top) vs CLEAR (bottom)",
                 y=1.01, fontsize=12)
    fig.text(0.5, -0.04,
             CAPTION + "\ncolor = fraction of each column's max, computed "
             "PER-PANEL (raw scales differ ~200x); cell text = raw score; "
             "grey = task not yet seen.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    save(fig, "fig4_retention_matrix_raw")


def _norm_matrix_figure(ref_c, ref_r, ref_name, fig_no, fname, note):
    """Normalized (retention) matrices, shared color scale across ALL four
    panels: rows = method (V5 / CLEAR), cols = order (canonical / reversed).
    """
    panels = [
        (MAT_C,       ref_c, ROWS_C, GAMES_C),
        (MAT_R,       ref_r, ROWS_R, GAMES_R),
        (MAT_CLEAR_C, ref_c, ROWS_C, GAMES_C),
        (MAT_CLEAR_R, ref_r, ROWS_R, GAMES_R),
    ]
    finite = np.concatenate([(M / r)[np.isfinite(M / r)]
                             for (M, r, _, _) in panels])
    vmin = min(finite.min(), 0.0)          # include negatives if present
    vmax = max(finite.max(), 1.0)          # include 1.0 (the reference line)
    # Keep 1.0 strictly inside (vmin, vmax) for TwoSlopeNorm.
    vmin = min(vmin, 0.999)
    vmax = max(vmax, 1.001)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    flat = axes.ravel()
    im = None
    for ax, (M, r, rows, cols) in zip(flat, panels):
        im = _draw_matrix(ax, M, rows, cols, "norm", ref=r, norm=norm, cmap=cmap)
    axes[0, 0].set_title("V5 (min-max) -- Canonical order")
    axes[0, 1].set_title("V5 (min-max) -- Reversed order")
    axes[1, 0].set_title("CLEAR -- Canonical order")
    axes[1, 1].set_title("CLEAR -- Reversed order")
    axes[0, 0].set_ylabel("training phase")
    axes[1, 0].set_ylabel("training phase")

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label(f"retention  (method / {ref_name})")
    cbar.ax.axhline(1.0, color="black", linewidth=1.0)  # 1.0 = matches ref

    fig.suptitle(f"Figure {fig_no} -- Forgetting matrix: retention vs "
                 f"{ref_name.upper()} (blue >= reference, red < reference; "
                 "white = 1.0); V5 (top) vs CLEAR (bottom)", y=1.01,
                 fontsize=12)
    fig.text(0.5, -0.04, CAPTION + "\n" + note, ha="center", fontsize=8,
             style="italic")
    save(fig, fname)


def figure5_vs_local():
    note = ("Denominator (Local specialist) is ORDER-DEPENDENT; diagonal < 1 "
            "means consolidation already trades off the just-learned game. "
            "Cells can be negative (Boxing canonical) or > 1.")
    _norm_matrix_figure(LOCAL_C, LOCAL_R, "Local", 5,
                        "fig5_retention_matrix_vs_local", note)


def figure6_vs_joint():
    note = ("Denominator (Joint ceiling) is FIXED / order-independent -> the "
            "cleaner comparison. Same shared color scale as Fig 5's panels.")
    _norm_matrix_figure(JOINT_C, JOINT_R, "Joint", 6,
                        "fig6_retention_matrix_vs_joint", note)


if __name__ == "__main__":
    figure1()
    rc_l, rr_l, mc_l, mr_l = figure2()
    rc_j, rr_j, mc_j, mr_j = figure3()
    figure4_raw()
    figure5_vs_local()
    figure6_vs_joint()

    print("\n=== Forgetting matrix -- CANONICAL (rows=after task, cols=game) ===")
    print("cols:", GAMES_C)
    for lbl, row in zip(ROWS_C, MAT_C):
        print(f"{lbl:16s}", " ".join("   .  " if not np.isfinite(v)
                                     else f"{v:7.1f}" for v in row))
    print("--- vs LOCAL (%) ---")
    for lbl, row in zip(ROWS_C, MAT_C / LOCAL_C):
        print(f"{lbl:16s}", " ".join("  .  " if not np.isfinite(v)
                                     else f"{v*100:5.0f}" for v in row))
    print("--- vs JOINT (%) ---")
    for lbl, row in zip(ROWS_C, MAT_C / JOINT_C):
        print(f"{lbl:16s}", " ".join("  .  " if not np.isfinite(v)
                                     else f"{v*100:5.0f}" for v in row))
    print("\n=== Forgetting matrix -- REVERSED ===")
    print("cols:", GAMES_R)
    for lbl, row in zip(ROWS_R, MAT_R):
        print(f"{lbl:16s}", " ".join("   .  " if not np.isfinite(v)
                                     else f"{v:7.1f}" for v in row))
    print("--- vs LOCAL (%) ---")
    for lbl, row in zip(ROWS_R, MAT_R / LOCAL_R):
        print(f"{lbl:16s}", " ".join("  .  " if not np.isfinite(v)
                                     else f"{v*100:5.0f}" for v in row))
    print("--- vs JOINT (%) ---")
    for lbl, row in zip(ROWS_R, MAT_R / JOINT_R):
        print(f"{lbl:16s}", " ".join("  .  " if not np.isfinite(v)
                                     else f"{v*100:5.0f}" for v in row))
    print()

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
