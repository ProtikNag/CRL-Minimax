"""Normalized task-restricted visualizations for the Atari v4 study + training curves.

Restricts the forgetting matrix / retention to the first ``--tasks`` tasks and
reports per-game NORMALIZED scores (norm = (raw - random) / (threshold - random);
random->0, threshold->1) so games on wildly different scales are comparable. Also
plots the per-iteration training reward and the normalized greedy learning curve
with task boundaries.

    python -m experiments.atari_v4_figures --seeds 0 4 --tasks 4
    python -m experiments.atari_v4_figures --seeds 0 4 --tasks 5
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
CCOL, FCOL = "#1b9e77", "#d95f02"  # constrained (green), finetune (orange)


def _norm(raw, game):
    r = RANDOM_SCORES[game]
    return (raw - r) / (THRESH[game] - r)


def _load(method, seed):
    d = f"results/atari5_ppo_v4_{method}_seed{seed}"
    return np.array(json.load(open(f"{d}/eval_matrix.json")))


def normalized_matrices(seeds, nt, games):
    out = {}
    for m in ("constrained", "finetune"):
        mats = []
        for s in seeds:
            M = _load(m, s)[:nt, :nt]
            N = np.array([[_norm(M[i, j], games[j]) for j in range(nt)] for i in range(nt)])
            mats.append(N)
        out[m] = np.array(mats)
    return out


def fig_forgetting_matrices(norm, out_dir, nt, games, sfx):
    fig, axes = plt.subplots(1, 2, figsize=(2.6 * nt + 1, 4.4))
    for ax, m, title in zip(axes, ("constrained", "finetune"),
                            ("Min-max (ours)", "Fine-tune (baseline)")):
        M = norm[m].mean(0)
        disp = M.copy()
        for i in range(nt):
            for j in range(nt):
                if j > i:
                    disp[i, j] = np.nan
        cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#dddddd")
        im = ax.imshow(disp, cmap=cmap, vmin=-0.1, vmax=1.1, aspect="auto")
        for i in range(nt):
            for j in range(nt):
                if j <= i:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(nt)); ax.set_xticklabels(games, rotation=30, ha="right")
        ax.set_yticks(range(nt)); ax.set_yticklabels([f"after T{i+1}" for i in range(nt)])
        ax.set_title(title, fontweight="600")
        ax.set_xlabel("evaluated on game")
    fig.suptitle(f"Normalized forgetting matrix, first {nt} tasks (random=0, threshold=1)\n"
                 "read DOWN a column = retention as later tasks train",
                 fontweight="600", y=1.04)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="normalized score")
    _save(fig, out_dir, f"forgetting_matrix_norm_{sfx}")


def fig_retention_curves(norm, out_dir, nt, games, sfx):
    fig, axes = plt.subplots(1, nt, figsize=(3.7 * nt, 3.6), sharey=True)
    for j, (ax, g) in enumerate(zip(axes, games)):
        xs = list(range(j + 1, nt + 1))
        for m, col, lab in (("constrained", CCOL, "min-max"), ("finetune", FCOL, "finetune")):
            mean = norm[m].mean(0)[:, j]
            sd = norm[m].std(0)[:, j]
            ys = [mean[i] for i in range(j, nt)]
            es = [sd[i] for i in range(j, nt)]
            ax.errorbar(xs, ys, yerr=es, marker="o", color=col, label=lab, capsize=3)
        ax.axhline(0, color="#999", lw=0.8, ls=":")
        ax.axhline(1, color="#999", lw=0.8, ls="--")
        ax.set_title(f"T{j+1}: {g}", fontweight="600")
        ax.set_xlabel("after task i"); ax.set_xticks(range(1, nt + 1))
        if j == 0: ax.set_ylabel("normalized score")
        if j == nt - 1: ax.legend(fontsize=8)
    fig.suptitle(f"Retention of each game as later tasks are learned (normalized, mean±std)\n"
                 "dotted=random(0), dashed=threshold(1)", fontweight="600", y=1.05)
    _save(fig, out_dir, f"retention_curves_norm_{sfx}")


def fig_final_bars(norm, out_dir, nt, games, sfx):
    fig, ax = plt.subplots(figsize=(1.7 * nt + 2, 4.2))
    x = np.arange(nt); w = 0.38
    for k, (m, col, lab) in enumerate((("constrained", CCOL, "min-max (ours)"),
                                       ("finetune", FCOL, "fine-tune"))):
        mean = norm[m].mean(0)[nt - 1, :]
        sd = norm[m].std(0)[nt - 1, :]
        ax.bar(x + (k - 0.5) * w, mean, w, yerr=sd, capsize=3, color=col, label=lab)
    ax.axhline(0, color="#999", lw=0.8, ls=":"); ax.axhline(1, color="#999", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(games)
    ax.set_ylabel(f"normalized score after task {nt}"); ax.legend()
    ax.set_title(f"Retention after {nt} tasks (normalized; dotted=random, dashed=threshold)",
                 fontweight="600")
    _save(fig, out_dir, f"retention_bars_norm_{sfx}")


def fig_training_curve(seed, out_dir):
    d = f"results/atari5_ppo_v4_constrained_seed{seed}"
    recs = [json.loads(l) for l in open(f"{d}/logs.jsonl")]
    train = {"task1", "local", "global", "finetune"}
    gt = {1: "Pong", 2: "Breakout", 3: "Boxing", 4: "Qbert", 5: "SpaceInvaders"}
    offset, last_key, lastep = 0, None, 0
    xs, ep, gnorm, gtask, bounds = [], [], [], [], {}
    for r in recs:
        ph = r.get("phase")
        if ph not in train or "step" not in r:
            continue
        key = (r.get("task"), ph)
        if last_key is not None and key != last_key:
            offset += lastep + 50
        it = offset + r["step"]; lastep = r["step"]; last_key = key
        t = r.get("task"); bounds.setdefault(t, it)
        xs.append(it); ep.append(r.get("ep_return_clipped"))
        g = r.get("greedy_score")
        if g is not None and t in gt:
            gnorm.append(_norm(g, gt[t])); gtask.append(it)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    a1.plot(xs, ep, color="#555", lw=1)
    a1.set_ylabel("clipped episode\nreturn (per iter)")
    a1.set_title(f"Training reward per iteration — constrained seed {seed}", fontweight="600")
    a2.plot(gtask, gnorm, color=CCOL, lw=1.2, marker=".", ms=3)
    a2.axhline(1, color="#999", ls="--", lw=0.8); a2.axhline(0, color="#999", ls=":", lw=0.8)
    a2.set_ylabel("normalized greedy\nscore (current game)")
    a2.set_xlabel("cumulative PPO iteration")
    for a in (a1, a2):
        for t, x in bounds.items():
            a.axvline(x, color="#bbb", lw=0.7)
    for t, x in bounds.items():
        a1.text(x, a1.get_ylim()[1], f" T{t}:{gt.get(t,'')}", fontsize=7,
                va="top", color="#333", rotation=90)
    _save(fig, out_dir, f"training_curve_constrained_seed{seed}")


def cl_metrics(norm, nt):
    def m(M):
        final = M[nt - 1, :nt]
        fg = [max(M[j:nt, j]) - M[nt - 1, j] for j in range(nt - 1)]
        bw = [M[nt - 1, j] - M[j, j] for j in range(nt - 1)]
        return float(np.mean(final)), float(np.mean(fg)), float(np.mean(bw))
    res = {}
    for meth in ("constrained", "finetune"):
        arr = np.array([m(M) for M in norm[meth]])
        res[meth] = {k: (arr[:, i].mean(), arr[:, i].std())
                     for i, k in enumerate(("avg_perf", "forgetting", "bwt"))}
    return res


def _save(fig, out_dir, name):
    for sub, ext in (("png", "png"), ("svg", "svg")):
        p = Path(out_dir) / sub; p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--tasks", type=int, default=4)
    args = ap.parse_args()
    nt = args.tasks
    games = ALL_GAMES[:nt]
    sfx = f"first{nt}" if nt < 5 else "all5"
    out = Path(f"reports/atari5_ppo_v4/figures/normalized_{sfx}")
    norm = normalized_matrices(args.seeds, nt, games)
    fig_forgetting_matrices(norm, out, nt, games, sfx)
    fig_retention_curves(norm, out, nt, games, sfx)
    fig_final_bars(norm, out, nt, games, sfx)
    for s in args.seeds:
        fig_training_curve(s, out)
    cm = cl_metrics(norm, nt)
    print(f"\n=== normalized CL metrics, first {nt} tasks (mean±std) ===")
    for meth in ("constrained", "finetune"):
        for k, (mu, sd) in cm[meth].items():
            print(f"  {meth:11s} {k:11s} {mu:6.3f} ± {sd:.3f}")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
