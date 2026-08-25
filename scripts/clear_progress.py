#!/usr/bin/env python3
"""Read-only progress dashboard for the CLEAR buffer-sweep runs.

Reads each run's logs.jsonl / eval_matrix.json / resource_usage.json (never
writes, never touches the running jobs) and prints, per sweep:
  * which task/phase it is on and step vs the frame-matched budget (% done),
  * the latest stop-eval greedy score (3-episode, vs threshold),
  * completed eval-matrix rows = greedy-100 retention as each task finishes.

Usage:  python scripts/clear_progress.py
"""
from __future__ import annotations
import json, glob, os

RESULTS = "results"
RUNS = [  # (dir suffix, label, snapshot_batches for footprint note)
    ("atari5_v5_clearA_equal_seed0", "A equal  (snap=4)", 4),
    ("atari5_v5_clearB_gen_seed0",   "B gen    (snap=16)", 16),
    ("atari5_v5_clearC_vgen_seed0",  "C vgen   (snap=48)", 48),
]
# clear per-task budget: task1 = task1_iters; tasks 2..k = local_iters+global_iters
TASK1_ITERS, PERTASK_ITERS = 1500, 6925
GAMES = ["Qbert", "Pong", "Breakout", "Boxing", "SpaceInvaders"]
THRESH = [4000, 18, 80, 55, 800]
TOTAL = TASK1_ITERS + 4 * PERTASK_ITERS  # 29,200


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _budget_before(task_idx1):  # cumulative iters before this 1-based task
    return 0 if task_idx1 == 1 else TASK1_ITERS + (task_idx1 - 2) * PERTASK_ITERS


def main():
    print(f"CLEAR buffer sweep — frame-matched budget {TOTAL:,} train iters/run "
          f"(ours V5 realized = 29,200)\n" + "=" * 78)
    for suffix, label, snap in RUNS:
        d = os.path.join(RESULTS, suffix)
        logs = _load_jsonl(os.path.join(d, "logs.jsonl"))
        if not logs:
            print(f"\n{label}: no logs yet (dir {'exists' if os.path.isdir(d) else 'MISSING'})")
            continue
        last = logs[-1]
        task, step, phase = last.get("task", 1), last.get("step", 0), last.get("phase", "?")
        done = _budget_before(task) + step
        pct = 100.0 * done / TOTAL
        game = GAMES[task - 1] if 1 <= task <= 5 else "?"
        print(f"\n{label}   [{phase}] task {task}/5 = {game}")
        print(f"   overall {done:,}/{TOTAL:,} iters ({pct:4.1f}%)")
        # latest stop-eval greedy (3-ep) per task
        gs = [x for x in logs if x.get("greedy_score") is not None]
        if gs:
            g = gs[-1]
            thr = THRESH[g["task"] - 1] if 1 <= g.get("task", 0) <= 5 else 0
            print(f"   latest stop-eval greedy: task {g['task']} "
                  f"{GAMES[g['task']-1]} = {g['greedy_score']:.1f} / thr {thr}")
        # completed eval-matrix rows (greedy-100 retention, written at task boundaries)
        em_path = os.path.join(d, "eval_matrix.json")
        if os.path.exists(em_path):
            em = json.load(open(em_path))
            print(f"   eval_matrix rows done: {len(em)}/5 (greedy-100 forgetting matrix)")
            for r, row in enumerate(em):
                cells = "  ".join(f"{GAMES[c][:4]}={v:.0f}" for c, v in enumerate(row))
                print(f"      after task {r+1}: {cells}")
    print("\n" + "=" * 78)
    print("footprint (stored frames, uint8 84x84x4 = 28KB each), reported side-by-side:")
    for _, label, snap in RUNS:
        fr = snap * 16 * 128 * 5
        print(f"   {label}: {fr:,} frames  (~{fr*28224/1e9:.1f} GB)   vs ours = 5 frozen models (~0.09 GB)")


if __name__ == "__main__":
    main()
