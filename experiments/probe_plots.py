"""Head-only consolidation probe: figures + verdict.

For a run trained with ppo.global_probe_head_only=true, this reads the normal
global-phase logs (V_k_global vs V_k_ref_local each phase) and the end-of-task
eval matrix, and produces, under a diagnostics/head_only_probe/<run>/ folder:

  * retention_matrix ..... normalized greedy retention (raw in parens)
  * current_task_VG_vs_VL  per global phase: does the frozen-trunk, head-only
                           consolidation KEEP the current-task value at the
                           local reference V_L?  (the decisive plot)

Verdict logic (printed):
  current task retained ~perfectly  => damage in the normal method is from
                                       SHARED-REPRESENTATION (trunk) updates.
  current task still degrades        => the CONSTRAINED OBJECTIVE itself
                                       compromises the current task.

    python -m experiments.probe_plots --run results/atari_probe_orderA_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from crl.envs.atari import RANDOM_SCORES
THR = {"Pong": 18, "Breakout": 50, "Boxing": 90, "Qbert": 2000, "SpaceInvaders": 600}


def _games(run):
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    return [t["game"] for t in cfg["env"]["tasks"]]


def _rows(run, phase):
    return [json.loads(l) for l in (run / "logs.jsonl").read_text().splitlines()
            if l.strip() and json.loads(l).get("phase") == phase]


def _norm(v, g):
    r = RANDOM_SCORES[g]
    return (v - r) / (THR[g] - r)


def retention_matrix(run, games, out):
    evals = [r["values"] for r in _rows(run, "eval")]
    nt, G = len(evals), len(games)
    M = np.full((nt, G), np.nan)
    for i, row in enumerate(evals):
        for j in range(min(len(row), G)):
            M[i, j] = _norm(row[j], games[j])
    disp = M.copy()
    for i in range(nt):
        for j in range(G):
            if j > i:
                disp[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * G, 1.4 + 0.9 * nt))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(disp, cmap=cmap, vmin=-0.1, vmax=1.1, aspect="equal")
    for i in range(nt):
        for j in range(G):
            if not np.isnan(disp[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}\n({evals[i][j]:.0f})", ha="center",
                        va="center", fontsize=8, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(G)); ax.set_xticklabels(games, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(nt)); ax.set_yticklabels([f"after T{i+1}" for i in range(nt)], fontsize=9)
    ax.set_xlabel("evaluated on game")
    ax.set_title("Head-only probe — normalized retention (greedy; raw in parens)\n"
                 f"order: {'>'.join(games)}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalized score")
    _save(fig, out, "retention_matrix")
    return M


def vg_vs_vl(run, games, out):
    """The decisive plot: V_k_global vs V_k_ref_local over each global phase."""
    g = _rows(run, "global")
    tasks = sorted({r["task"] for r in g})
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    verdict = []
    for ax, k in zip(axes[0], tasks):
        rows = [r for r in g if r["task"] == k and r.get("V_k_global") == r.get("V_k_global")]
        x = [r["step"] for r in rows]
        vg = [r["V_k_global"] for r in rows]
        vl = [r["V_k_ref_local"] for r in rows]
        ax.plot(x, vl, color="#333", ls="--", lw=2, label="V_L (local ref, frozen)")
        ax.plot(x, vg, color="#1b9e77", marker=".", label="V_G (global, head-only)")
        ax.set_title(f"task {k}: {games[k-1]}"); ax.set_xlabel("global iter")
        ax.set_ylabel("discounted value"); ax.legend(fontsize=8)
        if rows:
            gap = vl[-1] - vg[-1]
            frac = vg[-1] / vl[-1] if vl[-1] else float("nan")
            verdict.append((k, games[k-1], vl[-1], vg[-1], gap, frac))
    fig.suptitle("Head-only consolidation: does V_G reach the local reference V_L?",
                 fontweight="600", y=1.02)
    _save(fig, out, "current_task_VG_vs_VL")
    return verdict


def _save(fig, out, name):
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = Path(out) / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="diagnostics/head_only_probe")
    args = ap.parse_args()
    run = Path(args.run)
    games = _games(run)
    out = Path(args.out) / run.name
    retention_matrix(run, games, out)
    verdict = vg_vs_vl(run, games, out)
    print(f"\n[probe] {run.name}  order={'>'.join(games)}")
    print(f"  {'task':>16} {'V_L':>8} {'V_G':>8} {'gap':>8} {'V_G/V_L':>8}")
    for k, g, vl, vg, gap, frac in verdict:
        tag = "RETAINED" if frac >= 0.9 else ("partial" if frac >= 0.6 else "DEGRADED")
        print(f"  T{k} {g:>12} {vl:8.3f} {vg:8.3f} {gap:8.3f} {frac:8.2f}  {tag}")
    print(f"[probe] figures -> {out}")


if __name__ == "__main__":
    main()
