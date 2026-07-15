"""Continual-learning metrics for the Atari study (raw + normalized).

Given the per-seed forgetting matrices M (rows = state after finishing task i,
cols = game j), computes the standard continual-RL metrics reviewers expect,
both on raw game scores and on per-game **normalized** scores
``(raw - random) / (target - random)`` so scores across games (Pong ±21 vs
Qbert thousands) can be averaged meaningfully:

* Average Performance (AP)  -- mean final-row score over all games.
* Forgetting (F)            -- mean over earlier games of (peak - final), i.e.
                              how much was lost from each game's best.
* Backward Transfer (BWT)   -- mean over earlier games of (final - just-learned);
                              positive = the method kept improving old games.

Raw scores are always retained alongside the normalized ones for later checking.
"""

from __future__ import annotations

import numpy as np

from crl.envs.atari import RANDOM_SCORES


def normalize_matrix(M: np.ndarray, games: list[str], targets: list[float]) -> np.ndarray:
    """Per-game normalize a [rows, G] score matrix to (raw-random)/(target-random)."""
    out = np.full_like(M, np.nan, dtype=float)
    for j, (g, tgt) in enumerate(zip(games, targets)):
        rnd = RANDOM_SCORES.get(g, 0.0)
        denom = (tgt - rnd) if np.isfinite(tgt) and tgt != rnd else 1.0
        out[:, j] = (M[:, j] - rnd) / denom
    return out


def cl_metrics(M: np.ndarray) -> dict[str, float]:
    """Continual metrics from one [T, T(+)] matrix (rows=after task i, cols=game j).

    Uses the lower triangle (j <= i). T = number of trained tasks (rows).
    """
    T = M.shape[0]
    final = M[T - 1, :T]  # performance on every trained game at the end
    ap = float(np.mean(final))
    forgets, bwts = [], []
    for j in range(T - 1):  # earlier games only (the last has no "later" tasks)
        col = M[j:T, j]  # from when game j was learned (row j) to the end
        peak = float(np.max(col))
        just = float(M[j, j])  # score right after learning game j
        fin = float(M[T - 1, j])
        forgets.append(peak - fin)
        bwts.append(fin - just)
    return {
        "avg_performance": ap,
        "forgetting": float(np.mean(forgets)) if forgets else 0.0,
        "bwt": float(np.mean(bwts)) if bwts else 0.0,
    }


def summarize(mats: list[np.ndarray], games: list[str], targets: list[float]
              ) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean +/- std across seeds of the CL metrics, in raw and normalized space.

    ``mats`` = list of per-seed [T, G] matrices (same shape). Returns
    ``{"raw": {metric: (mean, std)}, "norm": {...}}``.
    """
    def stack(space_mats):
        rows = [cl_metrics(m) for m in space_mats]
        keys = rows[0].keys()
        return {k: (float(np.mean([r[k] for r in rows])),
                    float(np.std([r[k] for r in rows]))) for k in keys}

    raw = stack(mats)
    norm = stack([normalize_matrix(m, games, targets) for m in mats])
    return {"raw": raw, "norm": norm}
