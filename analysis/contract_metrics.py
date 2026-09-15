"""Metrics from a logging-contract run directory (docs/LOGGING_CONTRACT.md §6).

One reader for EVERY method (ours / finetune / clear / baseline / componet /
cka_rl), so the same numbers are computed the same way for all of them. It derives
everything from the contract artifacts in ``results/<run>/``:

* ``run.json``       -- reference {random, ceiling} for per-task normalization.
* ``eval_matrix.json`` -- lower-triangular raw end-of-task scores -> PERF, BWT,
                          forgetting, retention (via analysis.continual_metrics).
* ``progress.jsonl`` -- within-phase ``eval`` records (evaluated_on == current
                        task) give each task's learning-curve AUC -> FWT against a
                        paired ``method: baseline`` run; ``phase_end`` records give
                        compute (Σ wall_s_phase, Σ frames_phase).

FWT is ``null`` when no baseline run is supplied -- never estimated, per the
contract. Run: ``python -m analysis.contract_metrics results/<run> [--baseline results/<base>]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.continual_metrics import cl_metrics


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _matrix_from_eval_json(rows: list[list[float]], n: int) -> np.ndarray:
    """Pad the jagged lower-triangular eval_matrix (row i has i+1 entries) into a
    dense [T, T] array; unfilled future cells are NaN and never read by cl_metrics."""
    M = np.full((n, n), np.nan, dtype=float)
    for i, row in enumerate(rows):
        M[i, : len(row)] = row
    return M


def normalize(M: np.ndarray, random: list[float], ceiling: list[float]) -> np.ndarray:
    """(raw - random[j]) / (ceiling[j] - random[j]) per column, divide-by-zero safe."""
    out = np.full_like(M, np.nan, dtype=float)
    for j in range(M.shape[1]):
        rnd = random[j] if j < len(random) else 0.0
        cel = ceiling[j] if j < len(ceiling) else 1.0
        denom = (cel - rnd) if np.isfinite(cel) and cel != rnd else 1.0
        out[:, j] = (M[:, j] - rnd) / denom
    return out


def within_phase_auc(records: list[dict]) -> dict[int, float]:
    """Per task i, mean of the NORMALIZED within-phase eval scores (evaluated_on == i
    during task i's own local/task1 phase) -- the learning-curve AUC used for FWT."""
    buckets: dict[int, list[float]] = {}
    for r in records:
        if r.get("type") != "eval":
            continue
        # within-phase = the current task evaluating itself while it trains. The
        # discriminator is iter >= 0 (end-of-phase/diagonal rows use iter == -1),
        # NOT the phase label -- grow/finetune methods label their learning-curve
        # records with the method name (cka_rl/finetune/componet), which the old
        # phase filter wrongly excluded, nulling their FWT.
        if r.get("evaluated_on") != r.get("task_idx"):
            continue
        if r.get("iter", -1) < 0:
            continue
        val = r.get("normalized")
        if val is not None:
            buckets.setdefault(int(r["evaluated_on"]), []).append(float(val))
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def compute_metrics(run_dir: str | Path, baseline_dir: str | Path | None = None) -> dict:
    run_dir = Path(run_dir)
    run_json = json.loads((run_dir / "run.json").read_text())
    ref = run_json.get("reference", {})
    random = list(ref.get("random", []))
    ceiling = list(ref.get("ceiling", []))
    tasks = run_json.get("tasks", [])
    n = len(tasks) if tasks else len(random)

    eval_rows = json.loads((run_dir / "eval_matrix.json").read_text())
    n = n or len(eval_rows)
    M = _matrix_from_eval_json(eval_rows, n)
    Mn = normalize(M, random, ceiling)

    base = cl_metrics(Mn)  # avg_performance (=PERF), forgetting, bwt (lower triangle)
    final = Mn[n - 1, :n]
    metrics = {
        "run": str(run_dir),
        "n_tasks": n,
        "perf": float(np.nanmean(final)),
        "forgetting": base["forgetting"],
        "bwt": base["bwt"],
        "retention_per_task": [None if np.isnan(x) else float(x) for x in final],
    }

    prog = _read_jsonl(run_dir / "progress.jsonl")
    phase_ends = [r for r in prog if r.get("type") == "phase_end"]
    metrics["compute"] = {
        "wall_s_total": float(sum(r.get("wall_s_phase", 0.0) for r in phase_ends)),
        "frames_total": int(max((r.get("frames_total", 0) for r in prog), default=0)),
    }

    # FWT: mean over tasks of (AUC_i - AUC_i^b) / (1 - AUC_i^b). null without a baseline.
    auc = within_phase_auc(prog)
    if baseline_dir is not None and auc:
        base_prog = _read_jsonl(Path(baseline_dir) / "progress.jsonl")
        auc_b = within_phase_auc(base_prog)
        fts = []
        for i, a in auc.items():
            b = auc_b.get(i)
            if b is None:
                continue
            denom = 1.0 - b
            fts.append((a - b) / denom if abs(denom) > 1e-6 else 0.0)
        metrics["fwt"] = float(np.mean(fts)) if fts else None
        metrics["fwt_n_tasks"] = len(fts)
    else:
        metrics["fwt"] = None
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None, help="paired method:baseline run dir (for FWT)")
    args = ap.parse_args()
    print(json.dumps(compute_metrics(args.run_dir, args.baseline), indent=2))


if __name__ == "__main__":
    main()
