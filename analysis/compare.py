"""Cross-method comparison figures and the retention table.

Given two or more runs (typically the constrained method and the
unconstrained baseline) this builds the paper's core result figures:

    retention_curves   one panel per task, every method overlaid, vs step
                       (the paper's Fig 3), plus a mean-retention summary panel
    retention_bars     final per-task value, grouped by method
    retention_table    % of peak-observed ("expert") value retained at the end
                       (rendered figure + CSV; the paper's Tab 1)

Usage (programmatic; the driver experiments/compare_constraint.py calls this):
    build_comparison({"constrained": run_a, "unconstrained": run_b}, out_dir)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis.plots import load_records, task_boundaries
from analysis.style import AC, METHOD_COLORS, METHOD_LABELS, save_figure
import matplotlib.pyplot as plt


def _probe_arrays(run_dir: Path):
    records = load_records(run_dir)
    probes = [r for r in records if r.get("phase") == "probe"]
    steps = np.array([r["cumulative_step"] for r in probes])
    values = np.array([r["values"] for r in probes])  # [T_probe, num_tasks]
    return steps, values, probes


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, AC["blue"])


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _training_windows(boundaries, max_step):
    """Map task index (0-based) -> (train_start, train_end) cumulative step."""
    switch_steps = [s for s, _ in boundaries]
    windows = {}
    for idx, (step, task) in enumerate(boundaries):
        end = switch_steps[idx + 1] if idx + 1 < len(switch_steps) else max_step
        windows[task - 1] = (step, end)
    return windows


def plot_retention_curves(runs: dict[str, Path], out_dir: Path) -> None:
    """Per-task panels (each method overlaid) + a mean-retention summary."""
    data = {m: _probe_arrays(p) for m, p in runs.items()}
    any_probes = next(iter(data.values()))[2]
    num_tasks = len(any_probes[0]["values"])
    boundaries = task_boundaries(any_probes)
    max_step = max(s.max() for s, _, _ in data.values())
    windows = _training_windows(boundaries, max_step)

    fig, axes = plt.subplots(num_tasks + 1, 1, figsize=(9, 2.4 * (num_tasks + 1)),
                             sharex=True)

    def mark_boundaries(ax):
        ax.set_xlim(0, max_step)
        for step, task in boundaries:
            ax.axvline(step, color=AC["faint"], lw=1.0, ls="--", zorder=0)
            if step > 0:
                ax.text(step, ax.get_ylim()[1] * 0.94, f" T{task} starts",
                        fontsize=8, color=AC["muted"], ha="left", va="top")

    for task_idx in range(num_tasks):
        ax = axes[task_idx]
        ax.set_ylim(0, 1.0)
        # Shade this task's retention region (after it stops being current).
        _, train_end = windows.get(task_idx, (max_step, max_step))
        ax.axvspan(train_end, max_step, color=AC["green"], alpha=0.07, zorder=0)
        for method, (steps, values, _) in data.items():
            ax.plot(steps, values[:, task_idx], color=_method_color(method), lw=2.2,
                    label=_method_label(method), solid_capstyle="round")
        ax.set_ylabel(f"Task {task_idx + 1}\nvalue")
        mark_boundaries(ax)
        ax.set_title(f"Task {task_idx + 1}  (green band = retention region, "
                     f"task no longer trained)", fontsize=10, loc="left")
        if task_idx == 0:
            ax.legend(loc="center right", fontsize=9)

    # Summary panel: mean value across all tasks.
    ax = axes[-1]
    ax.set_ylim(0, 1.0)
    for method, (steps, values, _) in data.items():
        ax.plot(steps, values.mean(axis=1), color=_method_color(method), lw=2.4,
                label=_method_label(method))
    mark_boundaries(ax)
    ax.set_ylabel("Mean value\n(all tasks)")
    ax.set_title("Summary: mean value across all tasks", fontsize=10, loc="left")
    ax.set_xlabel("Cumulative primal steps across all phases")
    ax.legend(loc="center right", fontsize=9)

    fig.suptitle("Per-task retention: constrained min-max vs unconstrained baseline",
                 fontweight="600", y=0.995)
    fig.tight_layout()
    save_figure(fig, out_dir, "retention_curves")


def plot_retention_bars(runs: dict[str, Path], out_dir: Path) -> None:
    """Grouped bars of final per-task value by method."""
    finals = {m: _probe_arrays(p)[1][-1] for m, p in runs.items()}
    num_tasks = len(next(iter(finals.values())))
    methods = list(runs)
    x = np.arange(num_tasks)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * num_tasks, 4.5))
    for m_idx, method in enumerate(methods):
        offset = (m_idx - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, finals[method], width, color=_method_color(method),
                      label=_method_label(method))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9,
                    color=AC["text"])
    ax.set_xticks(x, [f"Task {i + 1}" for i in range(num_tasks)])
    ax.set_ylabel("Final value (end of full sequence)")
    ax.set_title("Final retained value per task")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(fig, out_dir, "retention_bars")


def build_retention_table(runs: dict[str, Path], out_dir: Path, tables_dir: Path) -> None:
    """Final value and % of the single-task 'expert' retained, per task/method.

    The expert reference for a task is the best value ANY method achieves on it
    over its run (a cross-method proxy for a single-task specialist). Retention
    = final / expert. This makes a method that never learns a task, or forgets
    it, show a low percentage instead of a misleading 100%.
    """
    per_method_final = {m: _probe_arrays(p)[1][-1] for m, p in runs.items()}
    per_method_peak = {m: _probe_arrays(p)[1].max(axis=0) for m, p in runs.items()}
    num_tasks = len(next(iter(per_method_final.values())))
    expert = np.array([
        max(per_method_peak[m][i] for m in runs) for i in range(num_tasks)
    ])

    rows = []
    for method in runs:
        final = per_method_final[method]
        retentions = [
            100.0 * final[i] / expert[i] if expert[i] > 0 else 0.0
            for i in range(num_tasks)
        ]
        for task_idx in range(num_tasks):
            rows.append({
                "method": method, "task": task_idx + 1,
                "final_value": round(float(final[task_idx]), 4),
                "expert_value": round(float(expert[task_idx]), 4),
                "retention_pct": round(retentions[task_idx], 1),
            })
        rows.append({
            "method": method, "task": "mean",
            "final_value": round(float(final.mean()), 4),
            "expert_value": round(float(expert.mean()), 4),
            "retention_pct": round(float(np.mean(retentions)), 1),
        })

    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "retention_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "task", "final_value", "expert_value", "retention_pct"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Rendered table figure.
    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.42 * len(rows)))
    ax.axis("off")
    header = ["Method", "Task", "Final value", "Expert value", "Retention %"]
    cells = [[_method_label(r["method"]).split(" (")[0], str(r["task"]),
              f"{r['final_value']:.3f}", f"{r['expert_value']:.3f}",
              f"{r['retention_pct']:.1f}%"] for r in rows]
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for col in range(len(header)):
        table[0, col].set_facecolor(AC["surface"])
        table[0, col].set_text_props(weight="600")
    for r_idx, r in enumerate(rows, start=1):
        if r["task"] == "mean":
            for col in range(len(header)):
                table[r_idx, col].set_facecolor("#EFF6FF")
                table[r_idx, col].set_text_props(weight="600")
    ax.set_title("Final retention table (% of peak-observed value retained)",
                 fontweight="600", pad=12)
    save_figure(fig, out_dir, "retention_table")


def build_performance_table(perf_by_method: dict, out_dir: Path, tables_dir: Path) -> None:
    """Concrete task performance (success rate, steps) from real rollouts.

    ``perf_by_method`` maps method -> list of crl.evaluation.Performance, one
    per task. This is the interpretable metric ("% of the task solved"), as
    opposed to the raw value the algorithm optimizes.
    """
    rows = []
    for method, perfs in perf_by_method.items():
        for task_idx, p in enumerate(perfs):
            rows.append({
                "method": method, "task": task_idx + 1,
                "success_rate_pct": round(p.success_rate * 100, 1),
                "mean_return": round(p.mean_return, 3),
                "mean_steps": round(p.mean_steps, 1),
            })

    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "performance_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "task", "success_rate_pct", "mean_return", "mean_steps"]
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.42 * (len(rows) + 1)))
    ax.axis("off")
    header = ["Method", "Task", "Success rate", "Mean return", "Mean steps"]
    cells = [[_method_label(r["method"]).split(" (")[0], str(r["task"]),
              f"{r['success_rate_pct']:.1f}%", f"{r['mean_return']:.3f}",
              f"{r['mean_steps']:.1f}"] for r in rows]
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for col in range(len(header)):
        table[0, col].set_facecolor(AC["surface"])
        table[0, col].set_text_props(weight="600")
    ax.set_title("Task performance from rollouts (what the policy actually does)",
                 fontweight="600", pad=12)
    fig.text(0.5, 0.01, "Success rate = fraction of episodes that reach the goal; "
             "mean steps = path length (lower is better). Value 0.83 = optimal here.",
             ha="center", fontsize=9, color=AC["muted"])
    save_figure(fig, out_dir, "performance_table")


def build_comparison(runs: dict[str, Path], out_dir: Path, tables_dir: Path) -> None:
    """Produce every comparison figure and the retention table/CSV."""
    plot_retention_curves(runs, out_dir)
    plot_retention_bars(runs, out_dir)
    build_retention_table(runs, out_dir, tables_dir)
