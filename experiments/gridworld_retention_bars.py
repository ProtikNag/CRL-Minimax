"""Normalized success-rate retention bars for the gridworld 20-task study.

Uses the RAW SCORE (goal-reaching success rate), not the discounted value: for a
goal-reaching task the success rate is inherently normalized (0 = never reaches
the goal, 1 = always). Reads the final-policy per-task success rates from the
committed ``reports/gridworld_20task/tables/performance_table.csv`` and plots
per-task grouped bars (min-max / local-free / fine-tune) with 95% CI.

    python -m experiments.gridworld_retention_bars
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAME = "gridworld_20task"
METHODS = [("constrained", "Min-max (ours)", "#1b9e77"),
           ("localfree", "Local-free (ours)", "#377eb8"),
           ("finetune", "Fine-tune (baseline)", "#d95f02")]


def main():
    table = Path(f"reports/{NAME}/tables/performance_table.csv")
    rows = list(csv.DictReader(open(table)))
    D: dict = {}
    for r in rows:
        D.setdefault(r["method"], {})[r["task"]] = (
            float(r["success_pct"]) / 100, float(r["success_ci"]) / 100)
    tasks = [str(t) for t in range(1, 21)] + ["mean"]

    fig, ax = plt.subplots(figsize=(17, 5))
    x = np.arange(len(tasks)); w = 0.26
    for k, (m, lab, col) in enumerate(METHODS):
        means = [D[m][t][0] for t in tasks]; cis = [D[m][t][1] for t in tasks]
        ax.bar(x + (k - 1) * w, means, w, yerr=cis, capsize=2, color=col,
               label=lab, edgecolor="white", linewidth=0.3)
    ax.axvline(19.5, color="#aaa", lw=1, ls="--")
    ax.axhline(1.0, color="#999", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9)
    ax.set_xlabel('task (1–20 gridworld navigation tasks; "mean" = average over all 20)')
    ax.set_ylabel("retained success rate\n(normalized: 1.0 = always reaches goal)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Retention after 20 tasks — normalized success rate (raw score) per task, "
                 f"{NAME}, mean ± 95% CI", fontweight="600")
    ax.legend(loc="lower left", ncol=3, framealpha=0.9)
    fig.tight_layout()
    out = Path(f"reports/{NAME}/figures")
    for sub, ext in (("png", "png"), ("svg", "svg")):
        p = out / sub; p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"retention_bars_success_norm.{ext}", dpi=130, bbox_inches="tight")
    print(f"saved {out}/{{png,svg}}/retention_bars_success_norm")
    for m, lab, _ in METHODS:
        print(f"  {lab}: mean retained success {D[m]['mean'][0]*100:.1f}%")


if __name__ == "__main__":
    main()
