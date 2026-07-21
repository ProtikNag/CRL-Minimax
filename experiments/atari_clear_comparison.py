"""Three-way comparison figures: min-max vs fine-tune vs CLEAR (Atari v4).

Normalized forgetting matrices (random=0, threshold=1) side by side, plus a
CL-metrics grouped bar chart (Average Performance / Forgetting / Backward
Transfer) for the three methods. Only methods with a completed eval_matrix are
plotted, so this can run as soon as the CLEAR seed finishes.

    python -m experiments.atari_clear_comparison --seeds 0
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

ALL_GAMES = ["Pong", "Breakout", "Boxing", "Qbert", "SpaceInvaders"]
THRESH = {"Pong": 18, "Breakout": 50, "Boxing": 90, "Qbert": 2000, "SpaceInvaders": 600}
# (method_key, label, color, run-name prefix). Full-budget min-max lives in the
# v5 runs (global_iters=3000); CLEAR is the v4 run. Fine-tune dropped per request.
METHODS = [("constrained", "Min-max (ours), full budget", "#1b9e77", "atari5_ppo_v5"),
           ("clear", "CLEAR (Rolnick'19)", "#7570b3", "atari5_ppo_v5")]


def _norm(raw, game):
    r = RANDOM_SCORES[game]
    return (raw - r) / (THRESH[game] - r)


def _load(method, seeds, nt, games, prefix="atari5_ppo_v4"):
    mats = []
    for s in seeds:
        p = Path(f"results/{prefix}_{method}_seed{s}/eval_matrix.json")
        if not p.exists():
            continue
        M = np.array(json.load(open(p)))
        if M.shape[0] < nt:
            continue
        M = M[:nt, :nt]
        mats.append(np.array([[_norm(M[i, j], games[j]) for j in range(nt)]
                              for i in range(nt)]))
    return np.array(mats) if mats else None


def cl_metrics(M, nt):
    final = M[nt - 1, :nt]
    fg = [max(M[j:nt, j]) - M[nt - 1, j] for j in range(nt - 1)]
    bw = [M[nt - 1, j] - M[j, j] for j in range(nt - 1)]
    return float(np.mean(final)), float(np.mean(fg)), float(np.mean(bw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tasks", type=int, default=5)
    args = ap.parse_args()
    nt = args.tasks
    games = ALL_GAMES[:nt]
    present = [(m, lab, col, pre) for m, lab, col, pre in METHODS
               if _load(m, args.seeds, nt, games, pre) is not None]
    if not present:
        raise SystemExit("no completed eval_matrices found")
    norms = {m: _load(m, args.seeds, nt, games, pre) for m, _, _, pre in present}

    out = Path("reports/atari5_ppo_v4/figures/clear_comparison")

    # --- forgetting matrices side by side --------------------------------
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.8))
    if len(present) == 1:
        axes = [axes]
    for ax, (m, lab, _, _pre) in zip(axes, present):
        M = norms[m].mean(0)
        disp = M.copy()
        for i in range(nt):
            for j in range(nt):
                if j > i:
                    disp[i, j] = np.nan
        cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#dddddd")
        im = ax.imshow(disp, cmap=cmap, vmin=-0.1, vmax=1.1, aspect="equal")
        for i in range(nt):
            for j in range(nt):
                if j <= i:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(nt)); ax.set_xticklabels(games, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(nt)); ax.set_yticklabels([f"after T{i+1}" for i in range(nt)], fontsize=8)
        ax.set_title(lab, fontweight="600")
        ax.set_xlabel("evaluated on game")
    fig.suptitle(f"Atari {nt}-task normalized forgetting matrix (random=0, threshold=1) — "
                 f"read DOWN a column = retention", fontweight="600", y=1.03)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="normalized score")
    _save(fig, out, "forgetting_matrix_3way")

    # --- CL metrics grouped bars -----------------------------------------
    labels = ["Avg Performance ↑", "Forgetting ↓", "Backward Transfer ↑"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(3); w = 0.8 / len(present)
    print(f"\n=== CL metrics, {nt} tasks (normalized, mean over seeds {args.seeds}) ===")
    for k, (m, lab, col, _pre) in enumerate(present):
        rows = np.array([cl_metrics(M, nt) for M in norms[m]])
        mean = rows.mean(0); sd = rows.std(0)
        ax.bar(x + (k - (len(present)-1)/2) * w, mean, w, yerr=sd, capsize=3,
               color=col, label=lab)
        print(f"  {lab:20s} AvgPerf {mean[0]:.3f}  Forgetting {mean[1]:.3f}  BWT {mean[2]:.3f}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("normalized"); ax.legend()
    ax.set_title(f"Continual-learning metrics — full-budget min-max vs CLEAR "
                 f"({nt} Atari tasks)", fontweight="600")
    _save(fig, out, "cl_metrics_3way")
    print(f"\nfigures -> {out}")


def _save(fig, out_dir, name):
    for sub, ext in (("png", "png"), ("svg", "svg")):
        p = Path(out_dir) / sub; p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


if __name__ == "__main__":
    main()
