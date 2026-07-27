"""Full visualization suite for a consolidation baseline run.

Reads results/<run>/logs.jsonl + expert_refs.json and produces, under
diagnostics/consolidation/<run>/{png,svg}/:

  retention_matrix ...... lower-triangular greedy retention heatmap (script:
                          consolidation_retention, run separately)
  retention_curves ...... per game, raw greedy score vs #tasks consolidated
                          (down-a-column view) with the fixed expert dashed
  final_retention_bars .. expert vs final global raw score per game (+ pct)
  consolidation_dynamics  per task: V_k_global vs V_k_ref_local over the global
                          phase, with mu on a twin axis (the constraint at work)
  mu_constraint ......... mu and F_G (squared shortfall) across all global steps
  ppo_health ............ entropy / approx_kl / clipfrac / pg_loss / v_loss

    python -m experiments.baseline_report --run results/consolidate10_seed0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(run: Path):
    rows = [json.loads(l) for l in (run / "logs.jsonl").read_text().splitlines()
            if l.strip()]
    refs = json.loads((run / "expert_refs.json").read_text())
    by = {}
    for r in rows:
        by.setdefault(r.get("phase", "?"), []).append(r)
    return by, refs


def _save(fig, out: Path, name: str):
    for ext in ("png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)


def retention_curves(by, refs, out):
    games = refs["games"]
    exp = refs["expert_scores"]
    ret = sorted(by.get("retention", []), key=lambda r: r["task"])
    G = len(games)
    ncol = 5
    nrow = (G + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow))
    axes = axes.ravel()
    for j, g in enumerate(games):
        ax = axes[j]
        xs, ys = [], []
        for r in ret:
            if j < len(r["global_scores"]):
                xs.append(r["task"])
                ys.append(r["global_scores"][j])
        ax.axhline(exp[j], ls="--", c="C1", lw=1.4, label="expert")
        ax.plot(xs, ys, "-o", c="C0", ms=4, label="global")
        ax.axvline(j + 1, c="0.7", lw=1, ls=":")  # when this game was learned
        ax.set_title(f"{g} (learned at T{j+1})", fontsize=9)
        ax.set_xlabel("after task #", fontsize=8)
        ax.set_ylabel("raw score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(G, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Retention curves — raw greedy score vs tasks consolidated "
                 "(dashed = fixed expert; dotted = task introduced)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, "retention_curves")


def final_retention_bars(by, refs, out):
    games = refs["games"]
    exp = np.array(refs["expert_scores"], float)
    ret = sorted(by.get("retention", []), key=lambda r: r["task"])
    last = ret[-1]
    glob = np.array(last["global_scores"], float)
    G = len(glob)
    x = np.arange(G)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(1.3 * G + 2, 7),
                                  gridspec_kw={"height_ratios": [2, 1]})
    w = 0.4
    ax.bar(x - w / 2, exp[:G], w, label="expert", color="C1")
    ax.bar(x + w / 2, glob, w, label="global (final)", color="C0")
    ax.set_yscale("symlog")
    ax.set_ylabel("raw score (symlog)")
    ax.set_xticks(x)
    ax.set_xticklabels(games[:G], rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_title(f"Final retention after all {len(games)} tasks — expert vs global")
    ax.grid(alpha=0.3, axis="y")
    pct = 100.0 * glob / np.where(exp[:G] != 0, exp[:G], np.nan)
    colors = ["C2" if p >= 60 else "C3" if p < 20 else "C0" for p in pct]
    ax2.bar(x, pct, color=colors)
    ax2.axhline(100, c="C1", ls="--", lw=1)
    ax2.set_ylabel("retention %")
    ax2.set_xticks(x)
    ax2.set_xticklabels(games[:G], rotation=30, ha="right", fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    for xi, p in zip(x, pct):
        ax2.text(xi, p + 3, f"{p:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    _save(fig, out, "final_retention_bars")


def consolidation_dynamics(by, refs, out):
    games = refs["games"]
    gl = by.get("global", [])
    tasks = sorted({r["task"] for r in gl})
    n = len(tasks)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for idx, t in enumerate(tasks):
        ax = axes[idx]
        rr = [r for r in gl if r["task"] == t]
        rr.sort(key=lambda r: r["step"])
        step = [r["step"] for r in rr]
        vg = [r["V_k_global"] for r in rr]
        vl = [r["V_k_ref_local"] for r in rr]
        ax.plot(step, vl, c="C1", lw=1.4, label="V_L (ref)")
        ax.plot(step, vg, c="C0", lw=1.4, label="V_G")
        ax.set_title(f"T{t}: {games[t-1]}", fontsize=9)  # global 'task' is 1-based (k=2..K)
        ax.set_xlabel("global step", fontsize=8)
        ax.set_ylabel("discounted V", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        axm = ax.twinx()
        axm.plot(step, [r["mu"] for r in rr], c="C3", lw=1.0, alpha=0.6)
        axm.set_ylabel("mu", color="C3", fontsize=8)
        axm.tick_params(labelsize=7, colors="C3")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Consolidation dynamics per task — V_G chasing V_L (ref); mu (red) is the dual",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, "consolidation_dynamics")


def mu_constraint(by, refs, out):
    gl = by.get("global", [])
    # global-phase order: task then step
    gl = sorted(gl, key=lambda r: (r["task"], r["step"]))
    x = np.arange(len(gl))
    mu = [r["mu"] for r in gl]
    fg = [r.get("F_G", np.nan) for r in gl]
    task = [r["task"] for r in gl]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(x, mu, c="C3", lw=0.9)
    a1.set_ylabel("mu (dual)")
    a1.grid(alpha=0.3)
    a1.set_title("Dual mu and constraint F_G across the whole run (vertical lines = new task)")
    a2.plot(x, fg, c="C0", lw=0.9)
    a2.set_ylabel("F_G (sq. shortfall)")
    a2.set_xlabel("global-phase iteration (concatenated across tasks)")
    a2.grid(alpha=0.3)
    # task boundaries
    for ax in (a1, a2):
        prev = task[0]
        for i, tk in enumerate(task):
            if tk != prev:
                ax.axvline(i, c="0.7", lw=0.8, ls=":")
                prev = tk
    fig.tight_layout()
    _save(fig, out, "mu_constraint")


def ppo_health(by, refs, out):
    gl = sorted(by.get("global", []), key=lambda r: (r["task"], r["step"]))
    x = np.arange(len(gl))
    task = [r["task"] for r in gl]
    metrics = [("entropy", "entropy"), ("approx_kl", "approx KL"),
               ("clipfrac", "clip fraction"), ("pg_loss", "policy loss"),
               ("v_loss", "value loss"), ("greedy_score", "current-task greedy score")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (k, lab) in zip(axes, metrics):
        y = [r.get(k, np.nan) for r in gl]
        ax.plot(x, y, lw=0.8, c="C0")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        prev = task[0]
        for i, tk in enumerate(task):
            if tk != prev:
                ax.axvline(i, c="0.7", lw=0.7, ls=":")
                prev = tk
    fig.suptitle("PPO optimization health across the run (dotted = task boundary)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out, "ppo_health")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="diagnostics/consolidation")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) / run.name
    by, refs = _load(run)
    retention_curves(by, refs, out)
    final_retention_bars(by, refs, out)
    consolidation_dynamics(by, refs, out)
    mu_constraint(by, refs, out)
    ppo_health(by, refs, out)
    print(f"[baseline_report] figures -> {out}/{{png,svg}}/")


if __name__ == "__main__":
    main()
