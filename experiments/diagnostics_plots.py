"""Global-phase diagnostics plots (from `global_diag` log rows).

Reads one run's logs.jsonl (produced with ppo.diagnostics=true) and, for each
global consolidation phase (task k >= 2), plots the quantities requested for
understanding the min-max dynamics:

  * V_k^L, V_k^G and their gap V_L - V_G .......... is the current-task value oscillating?
  * mu ........................................... dual multiplier trajectory
  * F_G vs eps ................................... constraint active / inactive / oscillating
  * ||g_new|| / ||g_old|| (+ raw norms) .......... which term drives the actor update
  * cos(g_new, g_old) ............................ do the two objectives align or conflict?
  * current-task greedy score .................... does consolidation drift off the new task?
  * each past task's value ....................... is one old task dominating / being forgotten?

Everything lands in a (non-versioned) diagnostics folder.

    python -m experiments.diagnostics_plots --run results/atari_diag_seed0 --out diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "logs.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("phase") == "global_diag"]


def _by_task(diag):
    tasks = {}
    for r in diag:
        tasks.setdefault(r["task"], []).append(r)
    for k in tasks:
        tasks[k].sort(key=lambda r: r["step"])
    return dict(sorted(tasks.items()))


def _series(rows, key):
    return [r[key] for r in rows]


def _short(name):
    return name.replace("atari-", "")


def _phase_figure(k, rows, out_dir: Path):
    x = _series(rows, "step")
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    cur = _short(rows[0].get("cur_name", f"task{k}"))
    fig.suptitle(f"Global (consolidation) phase — task {k}  [current task = {cur}]",
                 fontweight="600", y=1.0)

    # (0,0) V_L, V_G
    a = ax[0, 0]
    a.plot(x, _series(rows, "V_k_local"), label="V_k^L (local ref, fixed)", color="#333", ls="--")
    a.plot(x, _series(rows, "V_k_global"), label="V_k^G (global, est.)", color="#1b9e77", marker=".")
    a.set_title("Current-task value: local ref vs global"); a.set_xlabel("global iter")
    a.set_ylabel("discounted value"); a.legend(fontsize=8)

    # (0,1) gap V_L - V_G
    a = ax[0, 1]
    gap = _series(rows, "V_gap")
    a.plot(x, gap, color="#d95f02", marker=".")
    a.axhline(0, color="#999", lw=0.8)
    a.fill_between(x, gap, 0, where=[g > 0 for g in gap], color="#d95f02", alpha=0.15)
    a.set_title("V_L - V_G  (>0 = constraint violated; oscillating?)"); a.set_xlabel("global iter")

    # (0,2) mu
    a = ax[0, 2]
    a.plot(x, _series(rows, "mu"), color="#7570b3", marker=".")
    a.set_title("mu (dual multiplier)"); a.set_xlabel("global iter")

    # (0,3) F_G vs eps
    a = ax[0, 3]
    fg = _series(rows, "F_G"); eps = rows[0].get("eps", 0.0)
    a.plot(x, fg, color="#e7298a", marker=".", label="F_G (shortfall^2)")
    a.axhline(eps, color="#333", ls="--", lw=1, label=f"eps={eps:g}")
    a.set_title("Constraint F_G vs eps (active if >eps)"); a.set_xlabel("global iter")
    a.set_yscale("symlog", linthresh=max(eps, 1e-3)); a.legend(fontsize=8)

    # (1,0) grad ratio
    a = ax[1, 0]
    ratio = [min(r, 1e3) for r in _series(rows, "grad_ratio_new_over_old")]
    a.plot(x, ratio, color="#1b9e77", marker=".")
    a.axhline(1.0, color="#c00", ls="--", lw=1, label="parity (=1)")
    a.set_title("||g_new|| / ||g_old||  (<1 = old-task objective drives update)")
    a.set_xlabel("global iter"); a.set_yscale("log"); a.legend(fontsize=8)

    # (1,1) raw grad norms
    a = ax[1, 1]
    a.plot(x, _series(rows, "grad_norm_new"), color="#1b9e77", marker=".", label="||g_new|| (current)")
    a.plot(x, _series(rows, "grad_norm_old"), color="#7570b3", marker=".", label="||g_old|| (past)")
    a.set_title("Raw actor-gradient norms"); a.set_xlabel("global iter")
    a.set_yscale("log"); a.legend(fontsize=8)

    # (1,2) cos
    a = ax[1, 2]
    a.plot(x, _series(rows, "grad_cos_new_old"), color="#666", marker=".")
    a.axhline(0, color="#999", lw=0.8)
    a.set_ylim(-1.05, 1.05)
    a.set_title("cos(g_new, g_old)  (<0 = conflicting objectives)"); a.set_xlabel("global iter")

    # (1,3) current-task greedy + per-old-task greedy trajectories
    a = ax[1, 3]
    a.plot(x, _series(rows, "cur_greedy_score"), color="#000", lw=2, marker=".",
           label="current task (greedy)")
    # per-old-task value (normalized-free raw value) as separate lines
    names = [p["name"] for p in rows[0]["past"]]
    for name in names:
        ys = [next((p["greedy_score"] for p in r["past"] if p["name"] == name), np.nan)
              for r in rows]
        a.plot(x, ys, marker=".", alpha=0.8, label=f"{_short(name)} (old)")
    a.set_title("Performance during consolidation (greedy score)")
    a.set_xlabel("global iter"); a.legend(fontsize=7)

    fig.tight_layout()
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = out_dir / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"phase_task{k}.{ext}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase_task{k}")


def _per_old_task_value_figure(k, rows, out_dir: Path):
    """Dedicated panel: each old task's stochastic value V_i^G over the phase."""
    x = _series(rows, "step")
    names = [p["name"] for p in rows[0]["past"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in names:
        ys = [next((p["V_i_global"] for p in r["past"] if p["name"] == name), np.nan)
              for r in rows]
        ax.plot(x, ys, marker=".", label=_short(name))
    ax.set_title(f"Task {k} global phase — each old task's value V_i^G (is one dominating?)")
    ax.set_xlabel("global iter"); ax.set_ylabel("discounted value V_i^G"); ax.legend(fontsize=8)
    fig.tight_layout()
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = out_dir / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"old_task_values_task{k}.{ext}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved old_task_values_task{k}")


def _overview(tasks, out_dir: Path):
    """mu and grad-ratio across all phases on shared axes."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, rows in tasks.items():
        x = _series(rows, "step")
        ax[0].plot(x, _series(rows, "mu"), marker=".", label=f"task {k}")
        r = [min(v, 1e3) for v in _series(rows, "grad_ratio_new_over_old")]
        ax[1].plot(x, r, marker=".", label=f"task {k}")
    ax[0].set_title("mu per global phase"); ax[0].set_xlabel("global iter"); ax[0].legend(fontsize=8)
    ax[1].axhline(1.0, color="#c00", ls="--", lw=1)
    ax[1].set_title("||g_new||/||g_old|| per global phase"); ax[1].set_xlabel("global iter")
    ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    for sub, ext in (("png", "png"), ("svg", "svg")):
        d = out_dir / sub; d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"overview.{ext}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved overview")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run> dir with logs.jsonl")
    ap.add_argument("--out", default="diagnostics")
    args = ap.parse_args()
    run = Path(args.run)
    diag = _rows(run)
    if not diag:
        raise SystemExit(f"no 'global_diag' rows in {run}/logs.jsonl "
                         "(was ppo.diagnostics=true?)")
    tasks = _by_task(diag)
    out = Path(args.out) / run.name
    print(f"[diagnostics] {run.name}: {len(diag)} diag rows over phases {list(tasks)}")
    for k, rows in tasks.items():
        _phase_figure(k, rows, out)
        if rows[0]["past"]:
            _per_old_task_value_figure(k, rows, out)
    _overview(tasks, out)
    print(f"[diagnostics] figures -> {out}")


if __name__ == "__main__":
    main()
