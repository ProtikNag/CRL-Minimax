"""Summarize the trained experts: budget table + reward-curve gallery.

Reads every experts/<Game>/meta.json, writes a table (best greedy score, and the
frames/episodes/iters it took to get there -- the numbers needed to keep CLEAR
comparisons fair) and copies each reward curve into diagnostics/experts/ (which
is version-controlled, unlike the large model binaries under experts/).

    python -m experiments.experts_summary
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from crl.envs.atari import RANDOM_SCORES

OUT = Path("diagnostics/experts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(Path("experts").glob("*/")):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        g = m["game"]
        rows.append(m)
        for f in ("reward_curve.png", "meta.json", "train_log.csv", "config.json"):
            if (d / f).exists():
                (OUT / g).mkdir(parents=True, exist_ok=True)
                shutil.copy(d / f, OUT / g / f)

    if not rows:
        print("no completed experts yet"); return
    cols = ["game", "best_greedy", "random_score", "frames_at_best", "episodes_at_best",
            "iters_at_best", "total_frames", "total_episodes", "wall_s", "stop_reason"]
    with open(OUT / "experts_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for m in rows:
            w.writerow([m.get(c, "") for c in cols])

    lines = ["# Trained experts (Impala-CNN large, single-task, shared init)\n",
             "All greedy-100 scores (project rule). `frames_at_best` etc. are the training",
             "budget to reach the best model — use these to give CLEAR a matched budget.\n",
             "| game | best greedy | random | frames→best | episodes→best | iters→best | total frames | stop |",
             "|---|---|---|---|---|---|---|---|"]
    for m in sorted(rows, key=lambda r: r["game"]):
        fb = m.get("frames_at_best", 0) or 0
        lines.append(f"| {m['game']} | {m.get('best_greedy','?'):.0f} | "
                     f"{RANDOM_SCORES.get(m['game'],'?')} | {fb/1e6:.1f}M | "
                     f"{m.get('episodes_at_best','?')} | {m.get('iters_at_best','?')} | "
                     f"{(m.get('total_frames',0))/1e6:.1f}M | {m.get('stop_reason','?')} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"summarized {len(rows)} experts -> {OUT}")
    for m in sorted(rows, key=lambda r: r["game"]):
        print(f"  {m['game']:>14}: best greedy {m.get('best_greedy','?')}  "
              f"@ {(m.get('frames_at_best',0) or 0)/1e6:.1f}M frames  ({m.get('stop_reason')})")


if __name__ == "__main__":
    main()
