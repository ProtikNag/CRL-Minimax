"""Single-run diagnostic figures from one run directory.

Usage:
    python -m analysis.plots --run results/<run_dir> [--out <figures_dir>]

Produces (PNG under <out>/png, SVG under <out>/svg; default <out> is
<run_dir>/figures):
    learning_curves   global-policy value on every task vs cumulative step
    duals             lambda / mu trajectories against constraint values
    gaps              local-vs-global gap sequences (alternation cycling check)
    forgetting_matrix task x training-phase evaluation heatmap

These are diagnostics for a single run. Cross-method comparison figures
(the paper's Fig 3 / Tab 1) are produced by analysis/compare.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.style import AC, AC_SERIES, blue_sequential, save_figure
import matplotlib.pyplot as plt


def load_records(run_dir: Path) -> list[dict]:
    with open(run_dir / "logs.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_boundaries(probes: list[dict]) -> list[tuple[int, int]]:
    """(cumulative_step, new_task) at each task switch, from probe records."""
    boundaries, seen = [], None
    for record in probes:
        task = record["current_task"]
        if task != seen:
            boundaries.append((record["cumulative_step"], task))
            seen = task
    return boundaries


def plot_learning_curves(records: list[dict], out_dir: Path, title_suffix: str = "") -> None:
    """Global-policy value on each task vs cumulative training step.

    Vertical dashed lines mark task switches; shaded bands mark the region
    where a task is no longer the current task (its retention region).
    """
    probes = [r for r in records if r.get("phase") == "probe"]
    if not probes:
        return
    steps = np.array([r["cumulative_step"] for r in probes])
    num_tasks = len(probes[0]["values"])
    values = np.array([r["values"] for r in probes])  # [T_probe, num_tasks]
    boundaries = task_boundaries(probes)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, steps.max())
    for task_idx in range(num_tasks):
        ax.plot(steps, values[:, task_idx], color=AC_SERIES[task_idx % len(AC_SERIES)],
                lw=2.2, label=f"Task {task_idx + 1}", solid_capstyle="round")
    for step, task in boundaries:
        if step > 0:
            ax.axvline(step, color=AC["faint"], lw=1.0, ls="--", zorder=0)
            ax.text(step, 0.96, f" T{task} starts", fontsize=8, color=AC["muted"],
                    ha="left", va="top")

    ax.set_xlabel("Cumulative primal steps across all phases")
    ax.set_ylabel("Value of deployed (global) policy")
    ax.set_title(f"Per-task learning and retention{title_suffix}")
    ax.legend(title="Evaluated on", loc="center right", ncol=1)
    fig.text(0.5, -0.02,
             "The deployed global policy changes only during task-1 and global "
             "phases (flat during local phases). A retained task holds its value "
             "after its training ends.",
             ha="center", fontsize=9, color=AC["muted"])
    save_figure(fig, out_dir, "learning_curves")


def plot_duals(records: list[dict], out_dir: Path) -> None:
    """Dual multipliers and their constraint values, local and global phases."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    panels = [
        ("local", "lambda", "F_L", "Local phase: max multiplier $\\lambda$ and "
         "total squared shortfall $F_L$", axes[0]),
        ("global", "mu", "F_G", "Global phase: multiplier $\\mu$ and squared "
         "shortfall $F_G$", axes[1]),
    ]
    for phase, dual_key, cons_key, title, ax in panels:
        rows = [r for r in records if r.get("phase") == phase]
        if not rows:
            ax.set_visible(False)
            continue
        x = np.arange(len(rows))
        dual_symbol = r"$\lambda$" if dual_key == "lambda" else r"$\mu$"
        ax.plot(x, [r[dual_key] for r in rows], color=AC["blue"], lw=2.0,
                label=f"{dual_symbol} (dual)")
        ax.set_ylabel("dual multiplier", color=AC["blue"])
        ax.tick_params(axis="y", labelcolor=AC["blue"])
        twin = ax.twinx()
        twin.plot(x, [r[cons_key] for r in rows], color=AC["amber"], lw=1.5,
                  label=f"{cons_key} (constraint)")
        twin.set_ylabel("squared shortfall", color=AC["amber"])
        twin.tick_params(axis="y", labelcolor=AC["amber"])
        twin.grid(False)
        twin.spines["top"].set_visible(False)
        ax.set_title(title, fontsize=11)
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Logged primal steps (phases concatenated in order)")
    fig.suptitle("Dual dynamics", fontweight="600")
    fig.tight_layout()
    save_figure(fig, out_dir, "duals")


def plot_gaps(records: list[dict], out_dir: Path) -> None:
    """Local-vs-global value gaps per alternation cycle (cycling diagnostic)."""
    rows = [r for r in records if r.get("phase") == "gaps"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rows))
    ax.axhline(0.0, color=AC["faint"], lw=1.0, ls="--", zorder=0)
    ax.plot(x, [r["gap_current"] for r in rows], color=AC["blue"], lw=2.0,
            marker="o", ms=4, label="Current task  $V_k^{L}-V_k^{G}$")
    past_keys = sorted({k for r in rows for k in r if k.startswith("gap_past_")})
    for j, key in enumerate(past_keys):
        ax.plot(x, [r.get(key, np.nan) for r in rows],
                color=AC_SERIES[(j + 1) % len(AC_SERIES)], lw=1.6, marker="s", ms=3.5,
                label=f"Past task {int(key.split('_')[-1]) + 1}  $V^{{G}}-V^{{L}}$")
    ax.set_xlabel("Alternation cycle (all tasks concatenated)")
    ax.set_ylabel("Value gap")
    ax.set_title("Local vs global gap sequences (watch for growing oscillation)")
    ax.legend(loc="best", fontsize=9)
    save_figure(fig, out_dir, "gaps")


def plot_forgetting_matrix(matrix: list[list[float]], out_dir: Path) -> None:
    """Evaluation heatmap: rows = after task k, cols = evaluated task."""
    num_rows = len(matrix)
    num_cols = max(len(row) for row in matrix)
    grid = np.full((num_rows, num_cols), np.nan)
    for r, row in enumerate(matrix):
        grid[r, : len(row)] = row

    fig, ax = plt.subplots(figsize=(1.4 + 1.0 * num_cols, 1.2 + 0.8 * num_rows))
    image = ax.imshow(grid, cmap=blue_sequential(), aspect="auto", vmin=0.0)
    for r in range(num_rows):
        for c in range(num_cols):
            if not np.isnan(grid[r, c]):
                shade = AC["text"] if grid[r, c] < 0.6 * np.nanmax(grid) else "white"
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center",
                        fontsize=10, color=shade)
    ax.set_xticks(range(num_cols), [f"Task {i + 1}" for i in range(num_cols)])
    ax.set_yticks(range(num_rows), [f"after Task {k + 1}" for k in range(num_rows)])
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Training progress")
    ax.set_title("Forgetting matrix (global policy value)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="value", shrink=0.85)
    save_figure(fig, out_dir, "forgetting_matrix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory (results/<name>).")
    parser.add_argument("--out", default=None, help="Figures dir (default <run>/figures).")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "figures"
    records = load_records(run_dir)

    plot_learning_curves(records, out_dir)
    plot_duals(records, out_dir)
    plot_gaps(records, out_dir)
    matrix_path = run_dir / "eval_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as handle:
            plot_forgetting_matrix(json.load(handle), out_dir)
    print(f"[plots] {out_dir}/png and {out_dir}/svg written")


if __name__ == "__main__":
    main()
