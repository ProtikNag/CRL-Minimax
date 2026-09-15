"""Extract tidy, viz-ready tables from a logging-contract run directory.

The raw progress.jsonl already holds every number (within-phase FWT evals, every
end-of-phase forgetting-matrix row, dual traces, per-phase compute), but JSONL is
awkward to plot. This flattens it into long-format CSVs so detailed visualization
(forgetting heatmaps, per-task retention curves, learning curves, dual dynamics)
can be built locally without re-parsing the log. Raw progress.jsonl is still kept
alongside for anything these tables miss.

  python -m analysis.gridworld_viz_data results/<run> --out reports/<x>/<run>

Emits into --out: forgetting_matrix.csv, learning_curves.csv, duals.csv, phases.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analysis.contract_metrics import _read_jsonl, normalize
import numpy as np


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def extract(run_dir: str | Path, out_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rj = json.loads((run_dir / "run.json").read_text())
    ref = rj.get("reference", {})
    random, ceiling = list(ref.get("random", [])), list(ref.get("ceiling", []))
    tasks = rj.get("tasks", [])
    method = rj.get("method", "?")
    prog = _read_jsonl(run_dir / "progress.jsonl")

    def norm(raw, j):
        rnd = random[j] if j < len(random) else 0.0
        cel = ceiling[j] if j < len(ceiling) else 1.0
        d = (cel - rnd) if (np.isfinite(cel) and cel != rnd) else 1.0
        return (raw - rnd) / d

    # 1) Forgetting matrix (long): every end-of-task row × every seen task.
    rows = json.loads((run_dir / "eval_matrix.json").read_text())
    fm = []
    for i, row in enumerate(rows):                    # after finishing task i (0-based)
        for j, raw in enumerate(row):                 # score on task j
            fm.append([method, i, tasks[i] if i < len(tasks) else i,
                       j, tasks[j] if j < len(tasks) else j,
                       raw, norm(raw, j)])
    _write_csv(out / "forgetting_matrix.csv",
               ["method", "after_task", "after_task_name", "eval_task",
                "eval_task_name", "raw", "normalized"], fm)

    # 2) Within-phase learning curves (FWT): the training policy's greedy score on
    #    its OWN task over iters (evaluated_on == task_idx, iter >= 0).
    lc = []
    for r in prog:
        if r.get("type") == "eval" and r.get("iter", -1) >= 0 \
                and r.get("evaluated_on") == r.get("task_idx"):
            j = int(r["evaluated_on"])
            lc.append([method, j, r.get("evaluated_on_task"), r.get("phase"),
                       r["iter"], r.get("frames_total"), r["raw"],
                       r.get("normalized")])
    _write_csv(out / "learning_curves.csv",
               ["method", "task", "task_name", "phase", "iter", "frames_total",
                "raw", "normalized"], lc)

    # 3) Dual traces (ours only): mu / shortfall / coeff over iters.
    du = [[method, r.get("task_idx"), r.get("iter"), r.get("mu"),
           r.get("shortfall_current"), r.get("coeff_current"),
           r.get("grad_share_past"), r.get("t_wall")]
          for r in prog if r.get("type") == "dual"]
    if du:
        _write_csv(out / "duals.csv",
                   ["method", "task_idx", "iter", "mu", "shortfall_current",
                    "coeff_current", "grad_share_past", "t_wall"], du)

    # 4) Per-phase compute (iters / frames / wall).
    ph = [[method, r.get("task_idx"), r.get("task"), r.get("phase"),
           r.get("iters"), r.get("frames_phase"), r.get("wall_s_phase"),
           r.get("t_wall"), r.get("frames_total")]
          for r in prog if r.get("type") == "phase_end"]
    _write_csv(out / "phases.csv",
               ["method", "task_idx", "task", "phase", "iters", "frames_phase",
                "wall_s_phase", "t_wall", "frames_total"], ph)

    return {"forgetting_matrix": len(fm), "learning_curves": len(lc),
            "duals": len(du), "phases": len(ph)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(extract(args.run_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
