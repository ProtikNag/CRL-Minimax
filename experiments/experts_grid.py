"""One giant figure: training curve of every expert (reward vs frames).

Per game: greedy-100 score (reported) + smoothed training return, vs frames
(millions); a second x-axis annotation gives episodes/iters at the best model.

    python -m experiments.experts_grid
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crl.envs.atari import RANDOM_SCORES

GAMES = ["Pong", "Breakout", "Boxing", "Freeway", "SpaceInvaders",
         "Qbert", "Assault", "Krull", "Seaquest", "BeamRider"]


def _load(g):
    d = Path("experts") / g
    if not (d / "train_log.csv").exists():
        return None
    rows = list(csv.DictReader(open(d / "train_log.csv")))
    def col(n):
        return np.array([float(r[n]) if r[n] not in ("", "nan") else np.nan for r in rows])
    ev_f, ev_s = [], []
    for r in rows:
        if r["greedy_score"] not in ("", "nan"):
            ev_f.append(float(r["frames"]) / 1e6); ev_s.append(float(r["greedy_score"]))
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return dict(frames=col("frames") / 1e6, train=col("train_return_mean"),
                episodes=col("episodes"), ev_f=ev_f, ev_s=ev_s, meta=meta)


def main():
    ncol, nrow = 5, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow))
    axes = axes.flatten()
    for ax, g in zip(axes, GAMES):
        d = _load(g)
        if d is None:
            ax.set_title(f"{g} (no data)"); continue
        ax.plot(d["frames"], d["train"], color="#c9c9c9", lw=1.2,
                label="training return (smoothed)")
        ax.plot(d["ev_f"], d["ev_s"], color="#1b9e77", marker="o", ms=3,
                label="greedy-100 (reported)")
        r = RANDOM_SCORES.get(g)
        if r is not None:
            ax.axhline(r, color="#d62728", ls=":", lw=1, label="random")
        m = d["meta"]
        best = m.get("best_greedy"); fb = (m.get("frames_at_best", 0) or 0) / 1e6
        eb = m.get("episodes_at_best", "?")
        ax.set_title(f"{g}  —  best {best:.0f} @ {fb:.1f}M frames" if best is not None else g,
                     fontsize=10)
        ax.set_xlabel("frames (millions)"); ax.set_ylabel("raw game score")
        ax.text(0.02, 0.98, f"{m.get('total_frames',0)/1e6:.0f}M frames\n"
                f"{m.get('total_episodes','?')} episodes\n{eb} eps→best",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
        ax.legend(fontsize=6, loc="lower right")
    for ax in axes[len(GAMES):]:
        ax.axis("off")
    fig.suptitle("Single-task experts (Impala-CNN large) — training reward vs frames "
                 "(greedy-100 reported; grey = training return)", fontweight="600", y=1.01)
    fig.tight_layout()
    out = Path("diagnostics/experts")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"all_experts_training.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}/all_experts_training.png")


if __name__ == "__main__":
    main()
