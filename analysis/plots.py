"""Diagnostic figures from a run directory (always saved as PNG and SVG).

Usage:
    python -m analysis.plots --run results/<run_dir>

Produces, in ``<run_dir>/figures``:
    duals            lambda / mu trajectories with constraint values
    gaps             gap sequences across alternation cycles (cycling check)
    forgetting       task x training-phase evaluation matrix heatmap
    retention        per-task value curves over the task sequence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Academic figure defaults (Tufte spine, faint horizontal grid).
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E9ECEF",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#495057",
        "axes.labelcolor": "#212529",
        "xtick.color": "#6C757D",
        "ytick.color": "#6C757D",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

AC_SERIES = [
    "#2563EB", "#D97706", "#059669", "#DC2626",
    "#7C3AED", "#0891B2", "#BE185D", "#92400E",
]
AC_SURFACE = "#F8F9FA"


def _load_records(run_dir: Path) -> list[dict]:
    with open(run_dir / "logs.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Both formats, always."""
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.svg")
    plt.close(fig)


def plot_duals(records: list[dict], out_dir: Path) -> None:
    """Dual variables and constraint values over primal steps."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, phase, dual_key, cons_key, color_idx in (
        (axes[0], "local", "lambda", "F_L", 0),
        (axes[1], "global", "mu", "F_G", 1),
    ):
        rows = [r for r in records if r.get("phase") == phase]
        if not rows:
            continue
        steps = np.arange(len(rows))
        ax.plot(steps, [r[dual_key] for r in rows], color=AC_SERIES[color_idx],
                lw=1.8, label=dual_key)
        twin = ax.twinx()
        twin.plot(steps, [r[cons_key] for r in rows], color=AC_SERIES[3],
                  lw=1.2, alpha=0.7, label=cons_key)
        twin.axhline(0.0, color="#ADB5BD", lw=0.8, ls="--")
        twin.spines["top"].set_visible(False)
        ax.set_ylabel(dual_key)
        twin.set_ylabel(cons_key, color=AC_SERIES[3])
        ax.set_title(f"{phase} phase", fontsize=11)
    axes[1].set_xlabel("logged primal steps (concatenated phases)")
    _save(fig, out_dir, "duals")


def plot_gaps(records: list[dict], out_dir: Path) -> None:
    """Gap sequences per alternation cycle: the cycling diagnostic."""
    rows = [r for r in records if r.get("phase") == "gaps"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(rows))
    ax.plot(xs, [r["gap_current"] for r in rows], color=AC_SERIES[0], lw=1.8,
            marker="o", ms=3, label="current-task gap $V_k^{L}-V_k^{G}$")
    past_keys = sorted({k for r in rows for k in r if k.startswith("gap_past_")})
    for j, key in enumerate(past_keys):
        series = [r.get(key, np.nan) for r in rows]
        ax.plot(xs, series, color=AC_SERIES[(j + 1) % len(AC_SERIES)], lw=1.2,
                marker="s", ms=2.5, label=key.replace("gap_past_", "past gap i="))
    ax.set_xlabel("alternation cycle (all tasks concatenated)")
    ax.set_ylabel("value gap")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out_dir, "gaps")


def plot_forgetting_matrix(matrix: list[list[float]], out_dir: Path) -> None:
    """Evaluation matrix heatmap: rows = after task k, cols = evaluated task."""
    num_rows = len(matrix)
    num_cols = max(len(row) for row in matrix)
    grid = np.full((num_rows, num_cols), np.nan)
    for row_idx, row in enumerate(matrix):
        grid[row_idx, : len(row)] = row

    # Sequential colormap: surface white -> primary blue (no rainbow maps).
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ac_blue", [AC_SURFACE, AC_SERIES[0]]
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * num_cols, 1.0 + 0.7 * num_rows))
    image = ax.imshow(grid, cmap=cmap, aspect="auto")
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if not np.isnan(grid[row_idx, col_idx]):
                ax.text(col_idx, row_idx, f"{grid[row_idx, col_idx]:.2f}",
                        ha="center", va="center", fontsize=9, color="#212529")
    ax.set_xticks(range(num_cols), [f"task {i+1}" for i in range(num_cols)])
    ax.set_yticks(range(num_rows), [f"after {k+1}" for k in range(num_rows)])
    ax.set_xlabel("evaluated task")
    ax.set_ylabel("training progress")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="value", shrink=0.8)
    _save(fig, out_dir, "forgetting")


def plot_retention(matrix: list[list[float]], out_dir: Path) -> None:
    """Per-task value as training advances through the sequence."""
    num_tasks = max(len(row) for row in matrix)
    fig, ax = plt.subplots(figsize=(7, 4))
    for task_idx in range(num_tasks):
        xs, ys = [], []
        for after_k, row in enumerate(matrix, start=1):
            if task_idx < len(row):
                xs.append(after_k)
                ys.append(row[task_idx])
        ax.plot(xs, ys, color=AC_SERIES[task_idx % len(AC_SERIES)], lw=1.8,
                marker="o", ms=4, label=f"task {task_idx + 1}")
    ax.set_xlabel("tasks trained so far")
    ax.set_ylabel("value of global policy")
    ax.set_xticks(range(1, len(matrix) + 1))
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out_dir, "retention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory (results/<name>).")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    records = _load_records(run_dir)
    plot_duals(records, out_dir)
    plot_gaps(records, out_dir)
    matrix_path = run_dir / "eval_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as handle:
            matrix = json.load(handle)
        plot_forgetting_matrix(matrix, out_dir)
        plot_retention(matrix, out_dir)
    print(f"[plots] figures written to {out_dir} (png + svg)")


if __name__ == "__main__":
    main()
