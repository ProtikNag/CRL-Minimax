"""Aggregate the 3-seed Double-DQN min-max study into the report bundle.

Reads, for each seed:
    results/<name>_seed<s>/eval_matrix.json                 (ours: continual)
    results/<name>_finetune_seed<s>/eval_matrix.json        (naive baseline)
    results/<name>_expert_<game>_seed<s>/expert_score.json  (per-game ceiling)

Writes reports/<name>/:
    figures/per_game_panels_ci      per-game retention, own y-axis + ceiling (CI)
    figures/final_pct_of_best_ci    final score as % of best-possible (CI)
    figures/methods_comparison      DQN vs REINFORCE approaches (raw + %-of-ceiling)
    tables/dqn_scores.csv, methods_comparison.csv

Usage:
    python -m experiments.aggregate_dqn --name minatar_dqn_big --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GAMES = ["space_invaders", "breakout", "asterix", "seaquest"]
LABELS = ["SpaceInvaders", "Breakout", "Asterix", "Seaquest"]

# REINFORCE raw scores (final policy) from the earlier study, for comparison (d).
REINFORCE = {
    "REINFORCE constrained": [23.285, 1.500, 0.570, 0.890],
    "REINFORCE local-free": [28.560, 1.490, 0.390, 1.000],
}


def ci95(rows):
    """rows: [S, K] -> (mean[K], halfwidth[K]) at 95% (Student-t)."""
    a = np.asarray(rows, dtype=float)
    n = a.shape[0]
    m = a.mean(axis=0)
    if n < 2:
        return m, np.zeros_like(m)
    se = a.std(axis=0, ddof=1) / np.sqrt(n)
    return m, se * stats.t.ppf(0.975, n - 1)


def load_eval(results, run, seeds):
    """Stack final rows [S, 4] and keep full triangular matrices per seed."""
    mats, finals = [], []
    for s in seeds:
        p = Path(results) / f"{run}_seed{s}" / "eval_matrix.json"
        m = json.load(open(p))
        mats.append(m)
        finals.append(m[-1])
    return mats, np.array(finals)


def load_ceilings(results, name, seeds):
    rows = []
    for s in seeds:
        rows.append([json.load(open(
            Path(results) / f"{name}_expert_{g}_seed{s}" / "expert_score.json"
        ))["score"] for g in GAMES])
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="minatar_dqn_big")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    ours_mats, ours_final = load_eval(args.results_dir, args.name, args.seeds)
    ft_mats, ft_final = load_eval(args.results_dir, f"{args.name}_finetune", args.seeds)
    ceil_rows = load_ceilings(args.results_dir, args.name, args.seeds)

    out = Path(args.reports_dir) / args.name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    T = np.arange(1, 5)
    ceil_m, ceil_h = ci95(ceil_rows)
    om, oh = ci95(ours_final)
    fm, fh = ci95(ft_final)
    n = len(args.seeds)

    def game_curve(mats, gi):
        """mean,halfwidth of game gi across seeds for each t>=gi (else nan)."""
        means, hws = [], []
        for t in range(4):
            if gi <= t:
                vals = [mats[s][t][gi] for s in range(len(mats))]
                m, h = ci95(np.array(vals)[:, None])
                means.append(m[0]); hws.append(h[0])
            else:
                means.append(np.nan); hws.append(np.nan)
        return np.array(means), np.array(hws)

    # ---- Fig 1: per-game panels with CI + ceiling ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for gi, (ax, lab) in enumerate(zip(axes.flat, LABELS)):
        om_g, oh_g = game_curve(ours_mats, gi)
        fm_g, fh_g = game_curve(ft_mats, gi)
        ax.errorbar(T, om_g, yerr=oh_g, fmt="s-", color="#C44E52", lw=2,
                    capsize=3, label="Ours (min-max DDQN)")
        ax.errorbar(T, fm_g, yerr=fh_g, fmt="o--", color="#4C72B0", lw=2,
                    capsize=3, label="Fine-tune (naive)")
        ax.axhline(ceil_m[gi], color="#55A868", ls=":", lw=2,
                   label=f"Best possible ({ceil_m[gi]:.1f}±{ceil_h[gi]:.1f})")
        ax.axvline(gi + 1, color="k", alpha=.12, lw=6)
        ax.set_title(f"{lab}  (max reachable ≈ {ceil_m[gi]:.1f})")
        ax.set_xlabel("After learning task #"); ax.set_ylabel("Game score (raw)")
        ax.set_xticks(T); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"MinAtar continual via Double-DQN — per-game retention "
                 f"(mean ± 95% CI, {n} seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "figures" / "per_game_panels_ci.png", dpi=150)
    fig.savefig(out / "figures" / "per_game_panels_ci.svg")

    # ---- Fig 2: final score as % of best-possible ----
    opc = 100 * om / ceil_m; opc_h = 100 * oh / ceil_m
    fpc = 100 * fm / ceil_m; fpc_h = 100 * fh / ceil_m
    x = np.arange(4); w = 0.38
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x - w / 2, opc, w, yerr=opc_h, capsize=4, color="#C44E52",
            label="Ours (min-max DDQN)")
    ax2.bar(x + w / 2, fpc, w, yerr=fpc_h, capsize=4, color="#4C72B0",
            label="Fine-tune (naive)")
    ax2.axhline(100, color="#55A868", ls=":", lw=2, label="Best possible (100%)")
    for i in range(4):
        ax2.text(x[i] - w / 2, opc[i] + 2, f"{om[i]:.1f}", ha="center", fontsize=8)
        ax2.text(x[i] + w / 2, fpc[i] + 2, f"{fm[i]:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("% of best-possible score"); ax2.set_ylim(0, 130)
    ax2.set_title(f"Final policy: % of best-possible per game "
                  f"(raw score on bars; mean ± 95% CI, {n} seeds)")
    ax2.legend(); ax2.grid(axis="y", alpha=.3)
    fig2.tight_layout()
    fig2.savefig(out / "figures" / "final_pct_of_best_ci.png", dpi=150)
    fig2.savefig(out / "figures" / "final_pct_of_best_ci.svg")

    # ---- Fig 3 (d): DQN vs REINFORCE approaches ----
    methods = {
        "REINFORCE constrained": (REINFORCE["REINFORCE constrained"], None, "#8172B3"),
        "REINFORCE local-free": (REINFORCE["REINFORCE local-free"], None, "#CCB974"),
        "DQN min-max (ours)": (om, oh, "#C44E52"),
    }
    fig3, (axa, axb) = plt.subplots(1, 2, figsize=(15, 5.5))
    xm = np.arange(4); bw = 0.25
    # raw scores
    for j, (name, (vals, hw, col)) in enumerate(methods.items()):
        axa.bar(xm + (j - 1) * bw, vals, bw, yerr=hw, capsize=3, color=col, label=name)
    axa.plot(xm, ceil_m, "_", color="#55A868", markersize=22, markeredgewidth=3,
             label="DQN best-possible")
    axa.set_xticks(xm); axa.set_xticklabels(LABELS)
    axa.set_ylabel("Game score (raw)"); axa.set_title("Raw game score (different scales)")
    axa.legend(fontsize=8); axa.grid(axis="y", alpha=.3)
    # % of DQN ceiling (scale-fair)
    for j, (name, (vals, hw, col)) in enumerate(methods.items()):
        pc = 100 * np.array(vals) / ceil_m
        pch = 100 * np.array(hw) / ceil_m if hw is not None else None
        axb.bar(xm + (j - 1) * bw, pc, bw, yerr=pch, capsize=3, color=col, label=name)
    axb.axhline(100, color="#55A868", ls=":", lw=2, label="Best possible (100%)")
    axb.set_xticks(xm); axb.set_xticklabels(LABELS)
    axb.set_ylabel("% of DQN best-possible"); axb.set_title("Normalized to ceiling (scale-fair)")
    axb.legend(fontsize=8); axb.grid(axis="y", alpha=.3)
    fig3.suptitle("MinAtar: DQN min-max vs REINFORCE approaches (final policy)", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(out / "figures" / "methods_comparison.png", dpi=150)
    fig3.savefig(out / "figures" / "methods_comparison.svg")

    # ---- Tables ----
    with open(out / "tables" / "dqn_scores.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["game", "finetune", "finetune_ci", "ours", "ours_ci",
                      "best_possible", "best_ci", "ours_pct_of_best"])
        for i, g in enumerate(LABELS):
            wtr.writerow([g, f"{fm[i]:.2f}", f"{fh[i]:.2f}", f"{om[i]:.2f}",
                          f"{oh[i]:.2f}", f"{ceil_m[i]:.2f}", f"{ceil_h[i]:.2f}",
                          f"{opc[i]:.0f}%"])
        wtr.writerow(["MEAN", f"{fm.mean():.2f}", "", f"{om.mean():.2f}", "",
                      f"{ceil_m.mean():.2f}", "", f"{(100*om/ceil_m).mean():.0f}%"])
    with open(out / "tables" / "methods_comparison.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["method"] + LABELS + ["mean"])
        for name, vals in REINFORCE.items():
            wtr.writerow([name] + [f"{v:.2f}" for v in vals] + [f"{np.mean(vals):.2f}"])
        wtr.writerow(["DQN min-max (ours)"] + [f"{v:.2f}" for v in om] +
                     [f"{om.mean():.2f}"])
        wtr.writerow(["DQN best-possible"] + [f"{v:.2f}" for v in ceil_m] +
                     [f"{ceil_m.mean():.2f}"])

    print(f"[aggregate-dqn] {args.name}: {n} seeds")
    print(f"{'game':14s}{'finetune':>10}{'ours':>10}{'best':>8}{'ours%best':>10}")
    for i, g in enumerate(LABELS):
        print(f"{g:14s}{fm[i]:10.2f}{om[i]:10.2f}{ceil_m[i]:8.2f}{opc[i]:9.0f}%")
    print(f"{'MEAN':14s}{fm.mean():10.2f}{om.mean():10.2f}{ceil_m.mean():8.2f}"
          f"{(100*om/ceil_m).mean():9.0f}%")
    print(f"bundle -> {out}")


if __name__ == "__main__":
    main()
