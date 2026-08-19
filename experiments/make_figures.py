"""Regenerate the faithful continual-learning result figure set for one Atari run.

Self-contained: depends only on numpy + matplotlib (no seaborn, no repo-internal
modules) so a version-2 run can call it directly.

Reads:
    <run-dir>/figure_data.json            (authoritative; required)
    <run-dir>/expert_agreement.json       (optional; enables figure 5)

Writes into <out>/png and <out>/svg:
    fig1_forgetting_matrix        raw greedy-100 scores, lower-triangular
    fig2_expert_normalized        (score-random)/(expert-random), lower-triangular
    fig3_pct_expert_retention     score/expert*100 trajectory per task
    fig4a_avg_perf_over_tasks     mean expert-normalized perf over seen tasks
    fig4b_forgetting_bwt          final-minus-just-learned per task (raw + normalized)
    fig5_expert_agreement         windowed relative_gap matrix (if file present)

figure_data.json is authoritative for figures 1-4. figure_data.json and
expert_agreement.json may disagree on a per-game expert value (they are measured
separately); figure 5 uses expert_agreement.json's own numbers and is captioned
as such.

Usage:
    python experiments/make_figures.py \
        --run-dir results/atari4_v6_full_seed0 \
        --out reports/atari4_v6_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable


# ------------------------- house style ------------------------------------- #

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe categorical palette (Wong / Okabe-Ito).
WONG = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

DPI = 300


# ------------------------- IO ---------------------------------------------- #

def load_figure_data(run_dir: Path) -> dict:
    with open(run_dir / "figure_data.json") as f:
        d = json.load(f)
    games = list(d["games"])
    T = len(games)

    # Lower-triangular matrix -> dense TxT with NaN in the (unevaluated) upper triangle.
    M = np.full((T, T), np.nan, dtype=float)
    for k, row in enumerate(d["forgetting_matrix"]):
        for i, v in enumerate(row):
            if v is not None:
                M[k, i] = float(v)

    return {
        "games": games,
        "T": T,
        "M": M,
        "expert": np.asarray(d["expert_scores"], dtype=float),
        "random": np.asarray(d["random_scores"], dtype=float),
        "thresholds": np.asarray(d.get("thresholds", [np.nan] * T), dtype=float),
    }


def load_agreement(run_dir: Path):
    p = run_dir / "expert_agreement.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    games = list(d["games"])
    T = len(games)
    A = np.full((T, T), np.nan, dtype=float)
    for k, row in enumerate(d["agreement_matrix_relative_gap"]):
        for i, v in enumerate(row):
            if v is not None:
                A[k, i] = float(v)
    return {
        "games": games,
        "T": T,
        "A": A,
        "horizons": d.get("horizons"),
        "expert": d.get("expert_greedy_score"),
    }


# ------------------------- helpers ----------------------------------------- #

def expert_normalized(M, expert, random):
    """(score - random) / (expert - random): 1.0 = expert, 0 = random."""
    denom = (expert - random)
    denom = np.where(denom == 0, np.nan, denom)
    return (M - random[None, :]) / denom[None, :]


def _savefig(fig, out: Path, name: str):
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "png" / f"{name}.png", dpi=DPI)
    fig.savefig(out / "svg" / f"{name}.svg")
    plt.close(fig)
    return out / "png" / f"{name}.png"


def _lower_tri_labels(ax, games):
    ax.set_xticks(range(len(games)))
    ax.set_yticks(range(len(games)))
    ax.set_xticklabels(games, rotation=30, ha="right")
    ax.set_yticklabels([f"after {g}" for g in games])
    ax.set_xlabel("evaluated game")
    ax.set_ylabel("training stage")


# ------------------------- figure 1: raw matrix ---------------------------- #

def fig1_forgetting_matrix(data, out):
    games, M, T = data["games"], data["M"], data["T"]
    expert, random = data["expert"], data["random"]

    # Color by per-game expert-normalized value so different score scales are
    # comparable; annotate cells with the RAW score. Negative (below random)
    # values are shown honestly via a diverging map centered at 0.
    norm_vals = expert_normalized(M, expert, random)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = np.nanmax(np.abs(norm_vals[np.isfinite(norm_vals)]))
    vmax = max(vmax, 1.0)
    cmap = plt.get_cmap("RdBu")  # diverging, colorblind-distinguishable
    tsn = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    disp = np.ma.masked_invalid(norm_vals)
    im = ax.imshow(disp, cmap=cmap, norm=tsn)

    for k in range(T):
        for i in range(T):
            if i > k:
                continue  # upper triangle not evaluated
            raw = M[k, i]
            nv = norm_vals[k, i]
            # text color for contrast
            tc = "white" if (np.isfinite(nv) and abs(nv) > 0.55 * vmax) else "black"
            ax.text(i, k, f"{raw:.1f}", ha="center", va="center",
                    color=tc, fontsize=10)

    # Hatch the un-evaluated upper triangle so it reads as "not measured".
    for k in range(T):
        for i in range(T):
            if i > k:
                ax.add_patch(plt.Rectangle((i - 0.5, k - 0.5), 1, 1,
                             facecolor="0.9", edgecolor="0.75", hatch="//",
                             lw=0.0, zorder=1))

    _lower_tri_labels(ax, games)
    cbar = fig.colorbar(ScalarMappable(norm=tsn, cmap=cmap), ax=ax,
                        fraction=0.046, pad=0.04)
    cbar.set_label("cell color = expert-normalized  (0=random, 1=expert)")
    ax.set_title("Continual Atari — greedy-100 forgetting matrix\n"
                 "(cells = raw score; lower-triangular; upper = not evaluated)")
    return _savefig(fig, out, "fig1_forgetting_matrix")


# ------------------------- figure 2: normalized matrix --------------------- #

def fig2_expert_normalized(data, out):
    games, M, T = data["games"], data["M"], data["T"]
    expert, random = data["expert"], data["random"]
    N = expert_normalized(M, expert, random)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = max(np.nanmax(np.abs(N[np.isfinite(N)])), 1.0)
    cmap = plt.get_cmap("RdBu")
    tsn = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(np.ma.masked_invalid(N), cmap=cmap, norm=tsn)

    for k in range(T):
        for i in range(T):
            if i > k:
                ax.add_patch(plt.Rectangle((i - 0.5, k - 0.5), 1, 1,
                             facecolor="0.9", edgecolor="0.75", hatch="//",
                             lw=0.0, zorder=1))
                continue
            nv = N[k, i]
            tc = "white" if (np.isfinite(nv) and abs(nv) > 0.55 * vmax) else "black"
            ax.text(i, k, f"{nv:.2f}", ha="center", va="center",
                    color=tc, fontsize=10)

    _lower_tri_labels(ax, games)
    cbar = fig.colorbar(ScalarMappable(norm=tsn, cmap=cmap), ax=ax,
                        fraction=0.046, pad=0.04)
    cbar.set_label("expert-normalized score")
    ax.set_title("Expert-normalized performance\n"
                 r"$(\mathrm{score}-\mathrm{random})/(\mathrm{expert}-\mathrm{random})$"
                 "   (1=expert, 0=random, <0 worse than random)")
    return _savefig(fig, out, "fig2_expert_normalized")


# ------------------------- figure 3: % expert retention -------------------- #

def fig3_pct_expert_retention(data, out):
    """For each task i, its score across stages k>=i, as % of that task's expert."""
    games, M, T = data["games"], data["M"], data["T"]
    expert = data["expert"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, g in enumerate(games):
        stages = list(range(i, T))
        pct = [100.0 * M[k, i] / expert[i] for k in stages]
        ax.plot(stages, pct, marker="o", color=WONG[i % len(WONG)],
                label=f"{g} (expert {expert[i]:.0f})", lw=2)
        # mark the "just learned" point
        ax.scatter([i], [pct[0]], s=110, facecolor="none",
                   edgecolor=WONG[i % len(WONG)], linewidths=2, zorder=5)

    ax.axhline(100, color="0.4", ls="--", lw=1, label="expert (100%)")
    ax.axhline(0, color="0.7", ls=":", lw=1)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"after\n{g}" for g in games])
    ax.set_xlabel("continual-learning stage (task index at which score is measured)")
    ax.set_ylabel("score as % of that task's expert")
    ax.set_title("Percentage-of-expert retention across the task sequence\n"
                 "(open circle = just after that task was learned)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.margins(x=0.05)
    return _savefig(fig, out, "fig3_pct_expert_retention")


# ------------------------- figure 4a: avg perf over seen tasks ------------- #

def fig4a_avg_perf_over_tasks(data, out):
    games, M, T = data["games"], data["M"], data["T"]
    expert, random = data["expert"], data["random"]
    N = expert_normalized(M, expert, random)

    xs = list(range(1, T + 1))
    avg = [np.nanmean(N[k, : k + 1]) for k in range(T)]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, avg, marker="o", color=WONG[0], lw=2)
    for x, y in zip(xs, avg):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(1.0, color="0.4", ls="--", lw=1, label="expert level")
    ax.axhline(0.0, color="0.7", ls=":", lw=1, label="random level")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\n({games[k-1]})" for k in xs])
    ax.set_xlabel("number of tasks seen")
    ax.set_ylabel("mean expert-normalized score\nover seen tasks")
    ax.set_title("Average performance over seen tasks vs. sequence length")
    ax.legend(frameon=False, fontsize=9)
    return _savefig(fig, out, "fig4a_avg_perf_over_tasks")


# ------------------------- figure 4b: forgetting / BWT --------------------- #

def fig4b_forgetting_bwt(data, out):
    """Final score minus just-learned score, per task (raw and expert-normalized).

    Positive = performance dropped (forgetting). The last task has no forgetting
    (learned last), so it is omitted from the delta bars.
    """
    games, M, T = data["games"], data["M"], data["T"]
    expert, random = data["expert"], data["random"]
    N = expert_normalized(M, expert, random)

    idx = list(range(T - 1))  # exclude last task (no post-learning stage)
    raw_drop = [M[i, i] - M[T - 1, i] for i in idx]          # just-learned - final
    norm_drop = [N[i, i] - N[T - 1, i] for i in idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for ax, vals, ylab, ttl in [
        (ax1, raw_drop, "raw score dropped\n(just-learned - final)",
         "Forgetting (raw units)"),
        (ax2, norm_drop, "expert-normalized dropped",
         "Forgetting (expert-normalized)"),
    ]:
        colors = [WONG[3] if v > 0 else WONG[2] for v in vals]
        ax.bar(range(len(idx)), vals, color=colors, edgecolor="black", lw=0.6)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([games[i] for i in idx], rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        for x, v in enumerate(vals):
            ax.annotate(f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                        (x, v), textcoords="offset points",
                        xytext=(0, 6 if v >= 0 else -12), ha="center", fontsize=9)

    fig.suptitle("Forgetting per task  (final vs. just-learned; positive/orange = "
                 "dropped = forgetting, negative/green = improved = backward transfer; "
                 "last task excluded).\n"
                 "Note: raw panel is dominated by Qbert's ~1000x score scale — the "
                 "normalized panel is the honest cross-game comparison.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _savefig(fig, out, "fig4b_forgetting_bwt")


# ------------------------- figure 5: expert agreement ---------------------- #

def fig5_expert_agreement(agr, out):
    games, A, T = agr["games"], agr["A"], agr["T"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    # relative_gap in [0,1], lower=better. Reverse a sequential map so dark=good.
    cmap = plt.get_cmap("cividis_r")  # perceptually uniform, colorblind-safe
    norm = Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, norm=norm)

    for k in range(T):
        for i in range(T):
            if i > k:
                ax.add_patch(plt.Rectangle((i - 0.5, k - 0.5), 1, 1,
                             facecolor="0.9", edgecolor="0.75", hatch="//",
                             lw=0.0, zorder=1))
                continue
            v = A[k, i]
            if not np.isfinite(v):
                continue
            tc = "white" if v > 0.5 else "black"
            ax.text(i, k, f"{v:.2f}", ha="center", va="center",
                    color=tc, fontsize=10)

    _lower_tri_labels(ax, games)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("relative gap to expert (0 = matches/beats expert, lower=better)")
    ax.set_title("Windowed expert-agreement matrix\n"
                 "(relative_gap; lower-triangular; upper = not evaluated)")
    return _savefig(fig, out, "fig5_expert_agreement")


# ------------------------- main -------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing figure_data.json (+ optional expert_agreement.json)")
    ap.add_argument("--out", required=True, help="output directory for png/ and svg/")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_figure_data(run_dir)
    paths = []
    paths.append(fig1_forgetting_matrix(data, out))
    paths.append(fig2_expert_normalized(data, out))
    paths.append(fig3_pct_expert_retention(data, out))
    paths.append(fig4a_avg_perf_over_tasks(data, out))
    paths.append(fig4b_forgetting_bwt(data, out))

    agr = load_agreement(run_dir)
    if agr is not None:
        paths.append(fig5_expert_agreement(agr, out))
        print("[make_figures] expert_agreement.json found -> figure 5 written")
    else:
        print("[make_figures] no expert_agreement.json -> skipping figure 5")

    print("[make_figures] wrote:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
