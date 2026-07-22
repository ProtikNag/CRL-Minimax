"""EXPERIMENT 2 plot: is the scalar value constraint a sufficient statistic?

Reads a constrained run's global_diag rows and shows, per consolidation phase,
the current-task VALUE gap (V_L - V_G) against the BEHAVIORAL gap
KL(pi_local || pi_global) on current-task states. If the value gap -> 0 (the
constraint is satisfied) while the KL stays large, the global has matched the
local's *return* but not its *behavior* -> the scalar value constraint is too
weak (H2). Goes in diagnostics/value_constraint/<run>/.

    python -m experiments.value_constraint_plot --run results/exp2a_klmeasure_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="diagnostics/value_constraint")
    args = ap.parse_args()
    run = Path(args.run)
    rows = [json.loads(l) for l in (run / "logs.jsonl").read_text().splitlines()
            if l.strip() and json.loads(l).get("phase") == "global_diag"]
    if not rows or "kl_local_global" not in rows[0]:
        raise SystemExit("no KL-logged global_diag rows (needs diagnostics + local_policy)")
    tasks = sorted({r["task"] for r in rows})
    out = Path(args.out) / run.name

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.8 * len(tasks), 4.4), squeeze=False)
    for ax, k in zip(axes[0], tasks):
        rk = [r for r in rows if r["task"] == k]
        x = [r["step"] for r in rk]
        vgap = [max(0.0, r["V_gap"]) for r in rk]     # shortfall (value gap, >=0)
        kl = [r["kl_local_global"] for r in rk]
        ax.plot(x, vgap, color="#d95f02", marker=".", label="value gap  max(0, V_L−V_G)")
        ax.set_xlabel("global iter"); ax.set_ylabel("value gap", color="#d95f02")
        ax.tick_params(axis="y", labelcolor="#d95f02")
        ax2 = ax.twinx()
        ax2.plot(x, kl, color="#1b9e77", marker=".", label="KL(π_local ‖ π_global)")
        ax2.set_ylabel("behavioral KL", color="#1b9e77")
        ax2.tick_params(axis="y", labelcolor="#1b9e77")
        ax.set_title(f"task {k}")
    fig.suptitle("Experiment 2 — value gap vs behavioral gap: does V-gap→0 while KL stays large?"
                 "  (⇒ scalar value constraint too weak)", fontweight="600", y=1.03)
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = out / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"value_vs_behavior.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[exp2] {run.name}")
    for k in tasks:
        rk = [r for r in rows if r["task"] == k]
        print(f"  task {k} ({rk[0].get('cur_name','?').replace('atari-','')}): "
              f"value gap {max(0,rk[0]['V_gap']):.2f}→{max(0,rk[-1]['V_gap']):.2f} | "
              f"KL {rk[0]['kl_local_global']:.3f}→{rk[-1]['kl_local_global']:.3f}")
    print(f"[exp2] figure -> {out}")


if __name__ == "__main__":
    main()
