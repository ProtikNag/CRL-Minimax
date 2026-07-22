"""EXPERIMENT 1 (feasibility) plot: joint vs per-task ceiling.

Reads results/<run>/joint_result.json and shows, per game, the single-task
ceiling vs the joint (all-games-at-once) model, normalized (random=0, threshold=1).

Verdict: if joint >= ceiling on every game, a feasible shared theta EXISTS
(rules out infeasibility / capacity saturation) -> the sequential min-max is
leaving performance on the table (objective/optimization problem). If joint
falls well short on some game, capacity/feasibility is the binding constraint.

    python -m experiments.joint_plot --run results/exp1_joint_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crl.envs.atari import RANDOM_SCORES
_THR = {"Pong": 18, "Breakout": 50, "Boxing": 90, "Qbert": 2000, "SpaceInvaders": 600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="diagnostics/feasibility")
    args = ap.parse_args()
    run = Path(args.run)
    r = json.loads((run / "joint_result.json").read_text())
    games = [g.replace("atari-", "") for g in r["games"]]

    def nm(v, g):
        lo = RANDOM_SCORES[g]
        return (v - lo) / (_THR[g] - lo)

    ceil = [nm(v, g) for v, g in zip(r["ceilings"], games)]
    joint = [nm(v, g) for v, g in zip(r["joint"], games)]

    x = np.arange(len(games)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 + 1.5 * len(games), 4.8))
    b1 = ax.bar(x - w/2, ceil, w, color="#7570b3", label="single-task ceiling")
    b2 = ax.bar(x + w/2, joint, w, color="#1b9e77", label="joint (all games at once)")
    ax.axhline(1.0, color="#888", ls="--", lw=1, label="threshold")
    ax.axhline(0.0, color="#bbb", lw=0.8)
    for bars, raw in ((b1, r["ceilings"]), (b2, r["joint"])):
        for rect, rv in zip(bars, raw):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    f"{rv:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(games)
    ax.set_ylabel("normalized score (random=0, threshold=1)")
    ax.set_title("Experiment 1 — feasibility: joint vs single-task ceiling\n"
                 "(joint ≥ ceiling everywhere ⇒ a feasible shared θ exists)", fontsize=10)
    ax.legend()
    out = Path(args.out) / run.name
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = out / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"joint_vs_ceiling.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[exp1] {run.name}")
    print(f"  {'game':>14} {'ceiling':>9} {'joint':>9} {'joint/ceil':>11}")
    for g, c, j in zip(games, r["ceilings"], r["joint"]):
        lo = RANDOM_SCORES[g]
        frac = (j - lo) / (c - lo) if abs(c - lo) > 1e-6 else float("nan")
        verdict = "reaches" if frac >= 0.85 else ("partial" if frac >= 0.5 else "SHORT")
        print(f"  {g:>14} {c:9.1f} {j:9.1f} {frac:10.2f}  {verdict}")
    print(f"[exp1] figure -> {out}")


if __name__ == "__main__":
    main()
