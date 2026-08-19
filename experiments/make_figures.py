"""Regenerate the faithful continual-learning result figure set for one Atari run.

Self-contained: depends only on numpy + matplotlib (no seaborn, no repo-internal
modules) so a version-2 run can call it directly.

Reference model (Part A, "experts NOT stored")
-----------------------------------------------
The per-task reference is the run's own LOCAL model (the current-task
specialist / constraint target), NOT an external stored single-task expert.
Everything is normalized against that LOCAL reference. The word "expert" is
deliberately avoided as the reference label (it would be misleading for Part A).
The human-readable reference name is read from figure_data.json's
`reference_label` and is used verbatim in captions/axes.

Reads:
    <run-dir>/figure_data.json            (authoritative; required)
        keys: games, thresholds, forgetting_matrix (lower-triangular),
              reference_scores (per game), reference_label (string),
              random_scores. Backward-tolerant: falls back to legacy
              `expert_scores` if `reference_scores` is absent.
    <run-dir>/expert_agreement.json       (optional; enables figure 5)
        If present it is global-vs-LOCAL relative_gap; labeled
        "windowed agreement vs local model", lower=better. If absent
        (as for the v1 run), figure 5 is skipped gracefully.

Writes into <out>/png and <out>/svg:
    fig1_forgetting_matrix        raw greedy-100 scores, lower-triangular
    fig2_local_normalized         (score-random)/(reference-random), lower-triangular
    fig3_pct_local_retention      score/reference*100 trajectory per task
    fig4a_avg_perf_over_tasks     mean local-normalized perf over seen tasks
    fig4b_forgetting_bwt          just-learned - final per task (raw + normalized)
    fig5_local_agreement          windowed relative_gap matrix (if file present)

figure_data.json is authoritative for figures 1-4.

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

# Short label used in axis text / legends. The long provenance string from
# figure_data.json goes into the README, not onto the axes.
SHORT_REF = "local model"


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

    # Authoritative key is `reference_scores`; tolerate legacy `expert_scores`.
    if "reference_scores" in d:
        ref = np.asarray(d["reference_scores"], dtype=float)
        ref_label_long = d.get("reference_label", "local model")
    elif "expert_scores" in d:
        ref = np.asarray(d["expert_scores"], dtype=float)
        ref_label_long = d.get("reference_label", "local model")
    else:
        raise KeyError("figure_data.json has neither 'reference_scores' nor 'expert_scores'")

    return {
        "games": games,
        "T": T,
        "M": M,
        "reference": ref,
        "reference_label": ref_label_long,
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
    }


# ------------------------- helpers ----------------------------------------- #

def local_normalized(M, reference, random):
    """(score - random) / (reference - random): 1.0 = local model, 0 = random."""
    denom = (reference - random)
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
    reference, random = data["reference"], data["random"]

    # Color by per-game local-normalized value so different score scales are
    # comparable; annotate cells with the RAW score. Negative (below random)
    # values are shown honestly via a diverging map centered at 0.
    norm_vals = local_normalized(M, reference, random)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = np.nanmax(np.abs(norm_vals[np.isfinite(norm_vals)]))
    vmax = max(vmax, 1.0)
    cmap = plt.get_cmap("RdBu")  # diverging, colorblind-distinguishable
    tsn = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    disp = np.ma.masked_invalid(norm_vals)
    ax.imshow(disp, cmap=cmap, norm=tsn)

    for k in range(T):
        for i in range(T):
            if i > k:
                continue  # upper triangle not evaluated
            raw = M[k, i]
            nv = norm_vals[k, i]
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
    cbar.set_label(f"cell color = {SHORT_REF}-normalized  (0=random, 1={SHORT_REF})")
    ax.set_title("Continual Atari — greedy-100 forgetting matrix\n"
                 "(cells = raw score; lower-triangular; upper = not evaluated)")
    return _savefig(fig, out, "fig1_forgetting_matrix")


# ------------------------- figure 2: normalized matrix --------------------- #

def fig2_local_normalized(data, out):
    games, M, T = data["games"], data["M"], data["T"]
    reference, random = data["reference"], data["random"]
    N = local_normalized(M, reference, random)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = max(np.nanmax(np.abs(N[np.isfinite(N)])), 1.0)
    cmap = plt.get_cmap("RdBu")
    tsn = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ax.imshow(np.ma.masked_invalid(N), cmap=cmap, norm=tsn)

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
    cbar.set_label(f"{SHORT_REF}-normalized score")
    ax.set_title(f"{SHORT_REF.capitalize()}-normalized performance\n"
                 r"$(\mathrm{score}-\mathrm{random})/(\mathrm{local}-\mathrm{random})$"
                 f"   (1={SHORT_REF}, 0=random, >1 beats local, <0 worse than random)")
    return _savefig(fig, out, "fig2_local_normalized")


# ------------------------- figure 3: % local retention --------------------- #

def fig3_pct_local_retention(data, out):
    """For each task i, its score across stages k>=i, as % of that task's local model."""
    games, M, T = data["games"], data["M"], data["T"]
    reference = data["reference"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, g in enumerate(games):
        stages = list(range(i, T))
        pct = [100.0 * M[k, i] / reference[i] for k in stages]
        ax.plot(stages, pct, marker="o", color=WONG[i % len(WONG)],
                label=f"{g} ({SHORT_REF} {reference[i]:.1f})", lw=2)
        # mark the "just learned" point
        ax.scatter([i], [pct[0]], s=110, facecolor="none",
                   edgecolor=WONG[i % len(WONG)], linewidths=2, zorder=5)

    ax.axhline(100, color="0.4", ls="--", lw=1, label=f"{SHORT_REF} (100%)")
    ax.axhline(0, color="0.7", ls=":", lw=1)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"after\n{g}" for g in games])
    ax.set_xlabel("continual-learning stage (task index at which score is measured)")
    ax.set_ylabel(f"score as % of that task's {SHORT_REF}")
    ax.set_title(f"Percentage-of-{SHORT_REF} retention across the task sequence\n"
                 "(open circle = just after that task was learned; >100% beats local)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.margins(x=0.05)
    return _savefig(fig, out, "fig3_pct_local_retention")


# ------------------------- figure 4a: avg perf over seen tasks ------------- #

def fig4a_avg_perf_over_tasks(data, out):
    games, M, T = data["games"], data["M"], data["T"]
    reference, random = data["reference"], data["random"]
    N = local_normalized(M, reference, random)

    xs = list(range(1, T + 1))
    avg = [np.nanmean(N[k, : k + 1]) for k in range(T)]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(xs, avg, marker="o", color=WONG[0], lw=2)
    for x, y in zip(xs, avg):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(1.0, color="0.4", ls="--", lw=1, label=f"{SHORT_REF} level")
    ax.axhline(0.0, color="0.7", ls=":", lw=1, label="random level")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\n({games[k-1]})" for k in xs])
    ax.set_xlabel("number of tasks seen")
    ax.set_ylabel(f"mean {SHORT_REF}-normalized score\nover seen tasks")
    ax.set_title("Average performance over seen tasks vs. sequence length")
    ax.legend(frameon=False, fontsize=9)
    return _savefig(fig, out, "fig4a_avg_perf_over_tasks")


# ------------------------- figure 4b: forgetting / BWT --------------------- #

def fig4b_forgetting_bwt(data, out):
    """Just-learned score minus final score, per task (raw and local-normalized).

    Positive = performance dropped (forgetting). Negative = improved after the
    task was learned (backward transfer). The last task has no post-learning
    stage, so it is omitted from the delta bars.
    """
    games, M, T = data["games"], data["M"], data["T"]
    reference, random = data["reference"], data["random"]
    N = local_normalized(M, reference, random)

    idx = list(range(T - 1))  # exclude last task (no post-learning stage)
    raw_drop = [M[i, i] - M[T - 1, i] for i in idx]          # just-learned - final
    norm_drop = [N[i, i] - N[T - 1, i] for i in idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for ax, vals, ylab, ttl in [
        (ax1, raw_drop, "raw score dropped\n(just-learned - final)",
         "Forgetting (raw units)"),
        (ax2, norm_drop, f"{SHORT_REF}-normalized dropped",
         f"Forgetting ({SHORT_REF}-normalized)"),
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

    fig.suptitle("Forgetting per task  (just-learned - final; positive/orange = "
                 "dropped = forgetting, negative/green = improved = backward transfer; "
                 "last task excluded).\n"
                 "Note: raw panel is dominated by Qbert's ~1000x score scale — the "
                 f"{SHORT_REF}-normalized panel is the honest cross-game comparison. "
                 "Qbert's negative bar = backward transfer (global surpassed local).",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _savefig(fig, out, "fig4b_forgetting_bwt")


# ------------------------- figure 5: local agreement ----------------------- #

def fig5_local_agreement(agr, out):
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
    cbar.set_label(f"relative gap to {SHORT_REF} (0 = matches/beats local, lower=better)")
    ax.set_title(f"Windowed agreement vs {SHORT_REF}\n"
                 "(relative_gap; lower=better; lower-triangular; upper = not evaluated)")
    return _savefig(fig, out, "fig5_local_agreement")


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
    print(f"[make_figures] reference = {data['reference_label']}")
    paths = []
    paths.append(fig1_forgetting_matrix(data, out))
    paths.append(fig2_local_normalized(data, out))
    paths.append(fig3_pct_local_retention(data, out))
    paths.append(fig4a_avg_perf_over_tasks(data, out))
    paths.append(fig4b_forgetting_bwt(data, out))

    agr = load_agreement(run_dir)
    if agr is not None:
        paths.append(fig5_local_agreement(agr, out))
        print("[make_figures] expert_agreement.json found -> figure 5 (vs local) written")
    else:
        print("[make_figures] no expert_agreement.json -> skipping figure 5")

    print("[make_figures] wrote:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
