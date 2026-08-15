"""Confusion / forgetting matrix for the 4-game continual-Atari comparison.

Reads one or more run directories' ``eval_matrix.json`` (rows = state AFTER
finishing task i, cols = game j; ragged rows from ``eval_all_tasks: false`` are
padded with NaN) and renders the task x training-phase forgetting matrix as an
annotated heatmap, alongside the standard CL metrics (AP / Forgetting / BWT) in
raw and per-game-normalized space (``analysis.continual_metrics``).

Usage:
    python -m experiments.atari4_confusion \
        --runs results/atari4_stored_expert_seed0 results/atari4_minmax_seed0 \
        --labels "Stored-expert (B)" "Min-max (A)" \
        --out reports/atari4_confusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.continual_metrics import cl_metrics, normalize_matrix


def _load_square(run_dir: Path) -> tuple[np.ndarray, list[str], list[float]]:
    """Load eval_matrix.json into a square [T, T] matrix (NaN-padded), plus the
    game names and per-game targets (thresholds) read from the run's config."""
    rows = json.load(open(run_dir / "eval_matrix.json"))
    T = len(rows)
    M = np.full((T, T), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, v in enumerate(r[:T]):
            M[i, j] = v
    import yaml
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    tasks = cfg["env"].get("tasks") or cfg["env"]["params"].get("tasks")
    games = [t["game"] for t in tasks][:T]
    targets = [float(t.get("threshold", np.nan)) for t in tasks][:T]
    return M, games, targets


def _panel(ax, M: np.ndarray, games: list[str], title: str) -> None:
    T = M.shape[0]
    # Colour by per-column normalized score so games on different scales are comparable;
    # annotations still show the raw game score.
    im = ax.imshow(_col_normalize(M), cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(T)); ax.set_xticklabels(games, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels([f"after T{i+1}\n({games[i]})" for i in range(T)],
                                                fontsize=7)
    ax.set_xlabel("evaluated game"); ax.set_ylabel("training phase")
    ax.set_title(title, fontsize=10)
    for i in range(T):
        for j in range(T):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    return im


def _col_normalize(M: np.ndarray) -> np.ndarray:
    """Min-max normalize each column to [0,1] for colour only (annotations show raw)."""
    out = np.full_like(M, np.nan, dtype=float)
    for j in range(M.shape[1]):
        col = M[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        rng = (hi - lo) or 1.0
        out[:, j] = (col - lo) / rng
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs with eval_matrix.json")
    ap.add_argument("--labels", nargs="+", default=None, help="panel titles (one per run)")
    ap.add_argument("--out", default="reports/atari4_confusion", help="output dir (png+json)")
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    labels = args.labels or [r.name for r in runs]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), squeeze=False)
    summary = {}
    im = None
    for ax, run, label in zip(axes[0], runs, labels):
        M, games, targets = _load_square(run)
        raw = cl_metrics(M)
        norm = cl_metrics(normalize_matrix(M, games, targets))
        title = (f"{label}\nAP={raw['avg_performance']:.1f} "
                 f"F={raw['forgetting']:.1f} BWT={raw['bwt']:.1f}\n"
                 f"(norm) AP={norm['avg_performance']:.2f} "
                 f"F={norm['forgetting']:.2f} BWT={norm['bwt']:.2f}")
        im = _panel(ax, M, games, title)
        summary[label] = {"games": games, "matrix": M.tolist(),
                          "raw": raw, "norm": norm}
        print(f"\n=== {label} ({run.name}) ===")
        print("games:", games)
        print("matrix (rows=after Tk, cols=game):")
        for i, r in enumerate(M):
            print("  ", " ".join(f"{v:7.2f}" if np.isfinite(v) else "    nan" for v in r))
        print(f"RAW : AP={raw['avg_performance']:.2f}  "
              f"Forgetting={raw['forgetting']:.2f}  BWT={raw['bwt']:.2f}")
        print(f"NORM: AP={norm['avg_performance']:.3f}  "
              f"Forgetting={norm['forgetting']:.3f}  BWT={norm['bwt']:.3f}")

    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="per-column normalized score")
    fig.suptitle("Continual Atari (4 games) — forgetting / confusion matrix", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = out / "confusion_matrix.png"
    fig.savefig(png, dpi=150)
    json.dump(summary, open(out / "confusion_summary.json", "w"), indent=2)
    print(f"\n[confusion] wrote {png} and {out/'confusion_summary.json'}")


if __name__ == "__main__":
    main()
