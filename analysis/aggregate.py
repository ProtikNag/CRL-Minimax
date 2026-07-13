"""Multi-seed aggregation: mean +/- 95% CI figures and tables.

Given several seeds per method (each a self-contained run directory), this
builds the error-barred versions of the headline result:

    retention_curves_ci   per-task panels + mean-value summary, every method
                          overlaid as mean line + 95% CI band across seeds
    retention_bars_ci     final per-task value, grouped bars with 95% CI whiskers
    retention_table_ci    % of the expert value retained, mean +/- CI (fig + CSV)
    forgetting_matrix_mean seed-averaged task x training-phase heatmap
    forgetting_curve      average performance / forgetting vs number of tasks seen

Confidence intervals use the Student-t multiplier for the given seed count
(n=5 -> t_.975 = 2.776), which is the honest small-sample interval; with one
seed the band collapses to the point estimate.

Consumed by experiments/aggregate_seeds.py. All figures are written in PNG and
SVG through analysis.style.save_figure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis.plots import load_records, task_boundaries
from analysis.style import AC, METHOD_COLORS, METHOD_LABELS, save_figure
import matplotlib.pyplot as plt

# Sequential methods share an identical step schedule; joint is the upper bound.
SEQUENTIAL = ("constrained", "localfree", "finetune", "unconstrained")

# Student-t 0.975 quantile by seed count (two-sided 95% CI). Falls back to the
# normal 1.96 for larger n.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def _mean_ci(values: np.ndarray, axis: int = 0):
    """Return (mean, half-width of the 95% CI) along ``axis``."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    if n < 2:
        return mean, np.zeros_like(mean)
    sd = values.std(axis=axis, ddof=1)
    half = _T95.get(n, 1.96) * sd / np.sqrt(n)
    return mean, half


def _probe_stack(run_dirs: list[Path]):
    """Stack probe value arrays across seeds -> (steps, [S, P, T])."""
    steps_ref = None
    stack = []
    for run_dir in run_dirs:
        probes = [r for r in load_records(run_dir) if r.get("phase") == "probe"]
        steps = np.array([r["cumulative_step"] for r in probes])
        values = np.array([r["values"] for r in probes])  # [P, T]
        if steps_ref is None:
            steps_ref, min_len = steps, len(steps)
        else:
            min_len = min(len(steps_ref), len(steps))
            steps_ref = steps_ref[:min_len]
        stack = [s[:min_len] for s in stack] + [values[:min_len]]
    return steps_ref, np.stack(stack)  # [S, P, T]


def _final_stack(run_dirs: list[Path]) -> np.ndarray:
    """Final-row per-task value for each seed -> [S, T]."""
    rows = []
    for run_dir in run_dirs:
        with open(run_dir / "eval_matrix.json") as handle:
            rows.append(np.array(json.load(handle)[-1], dtype=float))
    return np.stack(rows)


def _forgetting_stack(run_dirs: list[Path]) -> np.ndarray:
    """Seed-averaged forgetting matrix -> ([S, R, C] with NaN padding)."""
    mats = []
    for run_dir in run_dirs:
        with open(run_dir / "eval_matrix.json") as handle:
            matrix = json.load(handle)
        num_cols = max(len(r) for r in matrix)
        grid = np.full((len(matrix), num_cols), np.nan)
        for r, row in enumerate(matrix):
            grid[r, : len(row)] = row
        mats.append(grid)
    return np.stack(mats)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def plot_retention_curves_ci(runs: dict[str, list[Path]], out_dir: Path,
                             metric_label: str = "value") -> None:
    """Per-task retention panels + summary, mean line and 95% CI band per method."""
    seq = {m: runs[m] for m in SEQUENTIAL if m in runs}
    data = {m: _probe_stack(dirs) for m, dirs in seq.items()}
    ref_steps = next(iter(data.values()))[0]
    num_tasks = next(iter(data.values()))[1].shape[2]

    # Task boundaries from any single run (deterministic schedule).
    probes = [r for r in load_records(next(iter(seq.values()))[0]) if r.get("phase") == "probe"]
    boundaries = task_boundaries(probes)
    max_step = int(ref_steps.max())
    # Adaptive y-limit: normalized scores sit near 1, but raw scores can exceed it.
    ymax = max(1.0, max(float(s.max()) for _, s in data.values()) * 1.08)

    ncols = 2
    nrows = int(np.ceil((num_tasks + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 2.5 * nrows),
                             sharex=True)
    axes = np.array(axes).reshape(-1)

    def mark(ax):
        ax.set_xlim(0, max_step)
        ax.set_ylim(0, ymax)
        for step, task in boundaries:
            if step > 0:
                ax.axvline(step, color=AC["faint"], lw=0.8, ls=":", zorder=0)

    for task_idx in range(num_tasks):
        ax = axes[task_idx]
        # Shade retention region: after this task stops being current.
        switch = [s for s, t in boundaries if t == task_idx + 2]
        train_end = switch[0] if switch else max_step
        ax.axvspan(train_end, max_step, color=AC["green"], alpha=0.06, zorder=0)
        for method, (steps, stack) in data.items():
            mean, half = _mean_ci(stack[:, :, task_idx])
            color = METHOD_COLORS.get(method, AC["blue"])
            ax.plot(steps, mean, color=color, lw=2.0, solid_capstyle="round",
                    label=METHOD_LABELS.get(method, method))
            ax.fill_between(steps, mean - half, mean + half, color=color, alpha=0.16,
                            lw=0)
        mark(ax)
        ax.set_title(f"Task {task_idx + 1}", fontsize=11, loc="left")
        ax.set_ylabel(metric_label)

    # Summary panel: mean value across tasks.
    ax = axes[num_tasks]
    for method, (steps, stack) in data.items():
        per_step = stack.mean(axis=2)  # [S, P]
        mean, half = _mean_ci(per_step)
        color = METHOD_COLORS.get(method, AC["blue"])
        ax.plot(steps, mean, color=color, lw=2.4, label=METHOD_LABELS.get(method, method))
        ax.fill_between(steps, mean - half, mean + half, color=color, alpha=0.16, lw=0)
    mark(ax)
    ax.set_title("Mean across all tasks", fontsize=11, loc="left")
    ax.set_ylabel(f"mean {metric_label}")

    for ax in axes[num_tasks + 1:]:
        ax.set_visible(False)
    for ax in axes[max(0, num_tasks - ncols + 1): num_tasks + 1]:
        ax.set_xlabel("Cumulative primal steps")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    n_seeds = next(iter(data.values()))[1].shape[0]
    fig.suptitle(f"Per-task retention  (mean ± 95% CI over {n_seeds} seeds; "
                 "shaded band = task no longer trained)",
                 fontweight="600", y=1.05)
    fig.tight_layout()
    save_figure(fig, out_dir, "retention_curves_ci")


def plot_retention_bars_ci(runs: dict[str, list[Path]], out_dir: Path,
                           metric_label: str = "value") -> None:
    """Grouped final-value bars with 95% CI whiskers, per task and method."""
    finals = {m: _final_stack(dirs) for m, dirs in runs.items()}
    num_tasks = next(iter(finals.values())).shape[1]
    methods = list(runs)
    x = np.arange(num_tasks)
    width = 0.8 / len(methods)
    n_seeds = next(iter(finals.values())).shape[0]

    fig, ax = plt.subplots(figsize=(2.0 + 1.15 * num_tasks, 4.8))
    for m_idx, method in enumerate(methods):
        mean, half = _mean_ci(finals[method])
        offset = (m_idx - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, mean, width, color=METHOD_COLORS.get(method, AC["blue"]),
               label=METHOD_LABELS.get(method, method),
               yerr=half, capsize=2.5, error_kw={"elinewidth": 1.0, "ecolor": AC["axis"]})
    ax.set_xticks(x, [f"T{i + 1}" for i in range(num_tasks)])
    ax.set_ylabel(f"Final {metric_label} (end of sequence)")
    ax.set_title(f"Final retained {metric_label} per task  (mean ± 95% CI, {n_seeds} seeds)",
                 loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    save_figure(fig, out_dir, "retention_bars_ci")


def plot_forgetting_matrix_mean(run_dirs: list[Path], out_dir: Path,
                                name: str = "forgetting_matrix_mean",
                                metric_label: str = "value") -> None:
    """Seed-averaged forgetting matrix: rows = after task k, cols = task."""
    stack = _forgetting_stack(run_dirs)  # [S, R, C]
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    num_rows, num_cols = mean.shape

    from analysis.style import blue_sequential
    fig, ax = plt.subplots(figsize=(1.6 + 0.72 * num_cols, 1.4 + 0.62 * num_rows))
    vmax = max(1.0, float(np.nanmax(mean)))
    image = ax.imshow(mean, cmap=blue_sequential(), aspect="auto", vmin=0.0, vmax=vmax)
    hi = np.nanmax(mean)
    for r in range(num_rows):
        for c in range(num_cols):
            if not np.isnan(mean[r, c]):
                shade = AC["text"] if mean[r, c] < 0.6 * hi else "white"
                ax.text(c, r, f"{mean[r, c]:.2f}", ha="center", va="center",
                        fontsize=8.5, color=shade)
    ax.set_xticks(range(num_cols), [f"T{i + 1}" for i in range(num_cols)])
    ax.set_yticks(range(num_rows), [f"after T{k + 1}" for k in range(num_rows)])
    ax.set_xlabel("Evaluated on task")
    ax.set_ylabel("Training progress")
    ax.set_title(f"Forgetting matrix (seed-mean {metric_label})")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=metric_label, shrink=0.85)
    save_figure(fig, out_dir, name)


def plot_average_performance_curve(runs: dict[str, list[Path]], out_dir: Path,
                                   metric_label: str = "value") -> None:
    """Average performance over tasks-seen-so-far vs number of tasks (CI band).

    For each method and each training milestone k (after task k finishes), the
    average value over the tasks seen so far -- the Continual-World 'average
    performance' curve. A flat-high curve means no forgetting as tasks accrue.
    """
    seq = {m: runs[m] for m in SEQUENTIAL if m in runs}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, dirs in seq.items():
        stack = _forgetting_stack(dirs)  # [S, R, C]
        num_rows = stack.shape[1]
        per_seed = np.array([
            [np.nanmean(stack[s, r, : r + 1]) for r in range(num_rows)]
            for s in range(stack.shape[0])
        ])  # [S, R]
        mean, half = _mean_ci(per_seed)
        ks = np.arange(1, num_rows + 1)
        color = METHOD_COLORS.get(method, AC["blue"])
        ax.plot(ks, mean, color=color, lw=2.2, marker="o", ms=4,
                label=METHOD_LABELS.get(method, method))
        ax.fill_between(ks, mean - half, mean + half, color=color, alpha=0.16, lw=0)
    ax.set_xlabel("Number of tasks trained so far")
    ax.set_ylabel(f"Average {metric_label} over tasks seen")
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
    ax.set_title("Average performance as the task sequence grows", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(fig, out_dir, "average_performance_curve")


def build_retention_table_ci(runs: dict[str, list[Path]], out_dir: Path,
                             tables_dir: Path) -> None:
    """Retention (% of expert) per task/method, mean +/- 95% CI. Figure + CSV."""
    finals = {m: _final_stack(dirs) for m, dirs in runs.items()}  # method -> [S, T]
    num_tasks = next(iter(finals.values())).shape[1]
    # Expert per task = best seed-mean value any method reaches on it.
    expert = np.array([
        max(finals[m][:, i].mean() for m in runs) for i in range(num_tasks)
    ])

    rows = []
    for method in runs:
        f = finals[method]  # [S, T]
        ret = 100.0 * f / np.where(expert > 0, expert, 1.0)  # [S, T]
        fmean, fhalf = _mean_ci(f)
        rmean, rhalf = _mean_ci(ret)
        # Per-seed task-mean, then CI, so the 'mean' row has an honest interval.
        seed_task_mean = f.mean(axis=1)
        seed_ret_mean = ret.mean(axis=1)
        for t in range(num_tasks):
            rows.append({"method": method, "task": t + 1,
                         "final_value": round(float(fmean[t]), 4),
                         "final_ci": round(float(fhalf[t]), 4),
                         "retention_pct": round(float(rmean[t]), 1),
                         "retention_ci": round(float(rhalf[t]), 1)})
        m_mean, m_half = _mean_ci(seed_task_mean)
        r_mean, r_half = _mean_ci(seed_ret_mean)
        rows.append({"method": method, "task": "mean",
                     "final_value": round(float(m_mean), 4),
                     "final_ci": round(float(m_half), 4),
                     "retention_pct": round(float(r_mean), 1),
                     "retention_ci": round(float(r_half), 1)})

    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "retention_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "task", "final_value", "final_ci", "retention_pct", "retention_ci"])
        writer.writeheader()
        writer.writerows(rows)

    # Rendered figure (mean +/- CI text).
    header = ["Method", "Task", "Final value", "Retention %"]
    cells = [[METHOD_LABELS.get(r["method"], r["method"]).split(" (")[0], str(r["task"]),
              f"{r['final_value']:.3f} ± {r['final_ci']:.3f}",
              f"{r['retention_pct']:.1f} ± {r['retention_ci']:.1f}%"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for col in range(len(header)):
        table[0, col].set_facecolor(AC["surface"])
        table[0, col].set_text_props(weight="600")
    for r_idx, r in enumerate(rows, start=1):
        if r["task"] == "mean":
            for col in range(len(header)):
                table[r_idx, col].set_facecolor("#EFF6FF")
                table[r_idx, col].set_text_props(weight="600")
    n_seeds = next(iter(finals.values())).shape[0]
    ax.set_title(f"Final retention table (% of expert retained, mean ± 95% CI, "
                 f"{n_seeds} seeds)", fontweight="600", pad=12)
    save_figure(fig, out_dir, "retention_table")


def plot_performance_bars_ci(values: dict[str, np.ndarray], out_dir: Path, *,
                             ylabel: str, title: str, filename: str,
                             reference: float | None = None,
                             reference_label: str | None = None) -> None:
    """Generic grouped-bar headline: per-task metric with 95% CI whiskers.

    ``values`` maps method -> [S, T]. Used for the interpretable performance
    metric (e.g. CartPole balancing steps, higher = better) rather than the raw
    value the algorithm optimizes.
    """
    num_tasks = next(iter(values.values())).shape[1]
    methods = list(values)
    x = np.arange(num_tasks)
    width = 0.8 / len(methods)
    n_seeds = next(iter(values.values())).shape[0]

    fig, ax = plt.subplots(figsize=(2.2 + 1.2 * num_tasks, 4.8))
    if reference is not None:
        ax.axhline(reference, color=AC["muted"], lw=1.2, ls="--", zorder=0,
                   label=reference_label or f"horizon ({reference:g})")
    for m_idx, method in enumerate(methods):
        mean, half = _mean_ci(values[method])
        offset = (m_idx - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, mean, width, color=METHOD_COLORS.get(method, AC["blue"]),
               label=METHOD_LABELS.get(method, method), yerr=half, capsize=2.5,
               error_kw={"elinewidth": 1.0, "ecolor": AC["axis"]})
    ax.set_xticks(x, [f"T{i + 1}" for i in range(num_tasks)])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (mean ± 95% CI, {n_seeds} seeds)", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    save_figure(fig, out_dir, filename)


def build_score_table_ci(returns: dict[str, np.ndarray], game_names: list[str],
                         out_dir: Path, tables_dir: Path,
                         random_scores: list[float] | None = None) -> None:
    """Actual game-score table (method x game, mean +/- 95% CI) + CSV.

    ``returns`` maps method -> [S, T] raw (unscaled) final-policy game scores.
    This is the paper's headline performance table for MinAtar.
    """
    methods = list(returns)
    num_games = next(iter(returns.values())).shape[1]
    n_seeds = next(iter(returns.values())).shape[0]

    rows = []
    for method in methods:
        r = returns[method]  # [S, T]
        mean, half = _mean_ci(r)
        seed_mean = r.mean(axis=1)
        mmean, mhalf = _mean_ci(seed_mean)
        rows.append({"method": method,
                     **{game_names[t]: (round(float(mean[t]), 2), round(float(half[t]), 2))
                        for t in range(num_games)},
                     "mean": (round(float(mmean), 2), round(float(mhalf), 2))})

    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "score_table.csv", "w", newline="") as handle:
        cols = ["method"] + game_names + ["mean"]
        writer = csv.writer(handle)
        writer.writerow(cols)
        if random_scores is not None:
            writer.writerow(["random"] + [round(s, 2) for s in random_scores]
                            + [round(float(np.mean(random_scores)), 2)])
        for r in rows:
            writer.writerow([METHOD_LABELS.get(r["method"], r["method"]).split(" (")[0]]
                            + [r[g][0] for g in game_names] + [r["mean"][0]])

    header = ["Method"] + game_names + ["Mean"]
    cells = []
    if random_scores is not None:
        cells.append(["Random"] + [f"{s:.2f}" for s in random_scores]
                     + [f"{np.mean(random_scores):.2f}"])
    for r in rows:
        cells.append([METHOD_LABELS.get(r["method"], r["method"]).split(" (")[0]]
                     + [f"{r[g][0]:.2f} ± {r[g][1]:.2f}" for g in game_names]
                     + [f"{r['mean'][0]:.2f} ± {r['mean'][1]:.2f}"])
    fig, ax = plt.subplots(figsize=(2.0 + 1.9 * (num_games + 1), 0.5 + 0.5 * (len(cells) + 1)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)
    for col in range(len(header)):
        table[0, col].set_facecolor(AC["surface"])
        table[0, col].set_text_props(weight="600")
    ax.set_title(f"Actual game score per task (raw mean return, mean ± 95% CI, "
                 f"{n_seeds} seed{'s' if n_seeds != 1 else ''})", fontweight="600", pad=12)
    save_figure(fig, out_dir, "score_table")


def build_performance_table_ci(success: dict[str, np.ndarray],
                               steps: dict[str, np.ndarray],
                               out_dir: Path, tables_dir: Path,
                               success_on_termination: bool = True) -> None:
    """Seed-averaged rollout performance table: success rate and path length.

    ``success`` / ``steps`` map method -> [S, T] arrays (per-seed, per-task
    success fraction and mean episode length). Rendered as mean ± 95% CI plus a
    CSV, alongside the value-based retention table.
    """
    methods = list(success)
    num_tasks = next(iter(success.values())).shape[1]
    n_seeds = next(iter(success.values())).shape[0]

    rows = []
    for method in methods:
        s_mean, s_half = _mean_ci(success[method])
        st_mean, st_half = _mean_ci(steps[method])
        for t in range(num_tasks):
            rows.append({"method": method, "task": t + 1,
                         "success_pct": round(float(s_mean[t] * 100), 1),
                         "success_ci": round(float(s_half[t] * 100), 1),
                         "mean_steps": round(float(st_mean[t]), 1),
                         "steps_ci": round(float(st_half[t]), 1)})
        sm, sh = _mean_ci(success[method].mean(axis=1))
        rows.append({"method": method, "task": "mean",
                     "success_pct": round(float(sm * 100), 1),
                     "success_ci": round(float(sh * 100), 1),
                     "mean_steps": round(float(steps[method].mean()), 1),
                     "steps_ci": 0.0})

    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "performance_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "task", "success_pct", "success_ci", "mean_steps", "steps_ci"])
        writer.writeheader()
        writer.writerows(rows)

    header = ["Method", "Task", "Success rate", "Mean steps"]
    cells = [[METHOD_LABELS.get(r["method"], r["method"]).split(" (")[0], str(r["task"]),
              f"{r['success_pct']:.1f} ± {r['success_ci']:.1f}%",
              f"{r['mean_steps']:.1f}"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.0, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for col in range(len(header)):
        table[0, col].set_facecolor(AC["surface"])
        table[0, col].set_text_props(weight="600")
    for r_idx, r in enumerate(rows, start=1):
        if r["task"] == "mean":
            for col in range(len(header)):
                table[r_idx, col].set_facecolor("#EFF6FF")
                table[r_idx, col].set_text_props(weight="600")
    ax.set_title(f"Task performance from rollouts (success rate, mean ± 95% CI, "
                 f"{n_seeds} seeds)", fontweight="600", pad=12)
    caption = ("Success rate = fraction of episodes reaching the goal; "
               "mean steps = path length (lower is better)."
               if success_on_termination else
               "Success rate = fraction of episodes surviving the full horizon; "
               "mean steps = balancing length (higher is better).")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color=AC["muted"])
    save_figure(fig, out_dir, "performance_table")
