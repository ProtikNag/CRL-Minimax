"""On-demand retention matrix for the consolidation run (no need to stop it).

Reads the retention rows logged after each consolidated task (global raw score
vs the fixed expert raw score per seen game) and builds the matrix UP TO whatever
task has been consolidated so far. Cell = ratio global/expert, annotated with the
raw global and expert scores. Read DOWN a column = how a game's retention evolves.

    python -m experiments.consolidation_retention --run results/consolidate10_seed0
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
    ap.add_argument("--out", default="diagnostics/consolidation")
    args = ap.parse_args()
    run = Path(args.run)
    rows = [json.loads(l) for l in (run / "logs.jsonl").read_text().splitlines()
            if l.strip() and json.loads(l).get("phase") == "retention"]
    if not rows:
        raise SystemExit("no retention rows yet (has task 1 finished?)")
    refs = json.loads((run / "expert_refs.json").read_text())
    games = refs["games"]
    G = len(games)
    nt = len(rows)  # tasks consolidated so far

    ratio = np.full((nt, G), np.nan)
    glob = np.full((nt, G), np.nan)
    exp = np.array(refs["expert_scores"])
    for r in rows:
        k = r["task"]
        for j in range(len(r["global_scores"])):
            glob[k - 1, j] = r["global_scores"][j]
            ratio[k - 1, j] = r["ratio"][j]

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * G, 1.5 + 0.85 * nt))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(ratio, cmap=cmap, vmin=0.0, vmax=1.1, aspect="equal")
    for i in range(nt):
        for j in range(G):
            if not np.isnan(ratio[i, j]):
                ax.text(j, i, f"{ratio[i,j]:.2f}\n{glob[i,j]:.0f}/{exp[j]:.0f}",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(G)); ax.set_xticklabels(games, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(nt)); ax.set_yticklabels([f"after T{i+1}" for i in range(nt)], fontsize=8)
    ax.set_xlabel("evaluated game (global raw / expert raw)")
    ax.set_title(f"Consolidation retention — {nt}/{G} tasks consolidated\n"
                 "cell = global/expert (greedy raw score); DOWN a column = retention",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="global / expert")
    out = Path(args.out) / run.name
    for ext in ("png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"retention_matrix.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[consolidate] {nt}/{G} tasks consolidated")
    print(f"  {'after':>8} " + " ".join(f"{g[:8]:>9}" for g in games))
    for i in range(nt):
        cells = [f"{ratio[i,j]:.2f}" if not np.isnan(ratio[i, j]) else "  ·  " for j in range(G)]
        print(f"  T{i+1:>6} " + " ".join(f"{c:>9}" for c in cells))
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
