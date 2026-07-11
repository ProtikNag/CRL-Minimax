"""Aggregate a multi-seed study into the seed-averaged report bundle.

Reads the raw run directories written by experiments/multiseed_comparison.py for
every (method, seed) and produces the error-barred headline figures + tables
under reports/<name>/ (tracked in git). Run this once all seeds have finished.

Bundle written to reports/<name>/:
    figures/{png,svg}/
        retention_curves_ci        per-task + summary, mean ± 95% CI
        retention_bars_ci          final per-task value, CI whiskers
        average_performance_curve  avg value over tasks-seen vs task count
        forgetting_matrix_mean     seed-averaged evaluation heatmap
        retention_table            % of expert retained, mean ± 95% CI
        performance_table          rollout success rate, mean ± 95% CI
        design_space_map / method_schematic   conceptual figures
        <method>/                  per-method diagnostics (seed 0): duals, gaps
    tables/
        retention_table.csv, performance_table.csv

Usage:
    python -m experiments.aggregate_seeds \
        --config configs/gridworld_tentask_sampled.yaml \
        --name gridworld_tentask --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.aggregate import (
    SEQUENTIAL, build_performance_table_ci, build_retention_table_ci,
    plot_average_performance_curve, plot_forgetting_matrix_mean,
    plot_performance_bars_ci, plot_retention_bars_ci, plot_retention_curves_ci,
)
from analysis.plots import load_records, plot_duals, plot_gaps, plot_forgetting_matrix
from crl.config import load_config
from crl.envs import make_family
from crl.evaluation import rollout_performance
from crl.policies import make_policy

ALL_METHODS = ("constrained", "unconstrained", "finetune", "joint")


def _discover(results_dir: Path, name: str, seeds: list[int],
              methods: tuple[str, ...]) -> dict[str, list[Path]]:
    """{method: [existing run dirs, one per seed]} (skips missing runs)."""
    runs: dict[str, list[Path]] = {}
    for method in methods:
        dirs = []
        for seed in seeds:
            path = results_dir / f"{name}_{method}_seed{seed}"
            if (path / "eval_matrix.json").exists():
                dirs.append(path)
            else:
                print(f"[aggregate] WARNING missing run: {path}")
        if dirs:
            runs[method] = dirs
    return runs


def _performance_stacks(config, runs: dict[str, list[Path]], num_episodes: int):
    """Roll out each final policy; return per-method [S, T] success/steps arrays."""
    family = make_family(config.env)
    success: dict[str, np.ndarray] = {}
    steps: dict[str, np.ndarray] = {}
    for method, dirs in runs.items():
        s_rows, st_rows = [], []
        for run_dir in dirs:
            policy = make_policy(config.policy, family)
            policy.load_state_dict(torch.load(run_dir / "final_policy.pt",
                                              weights_only=True))
            policy.eval()
            perfs = [rollout_performance(policy, task, num_episodes)
                     for task in family.tasks]
            s_rows.append([p.success_rate for p in perfs])
            st_rows.append([p.mean_steps for p in perfs])
        success[method] = np.array(s_rows)
        steps[method] = np.array(st_rows)
    return success, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--perf-episodes", type=int, default=200)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS),
                        choices=ALL_METHODS,
                        help="Subset of methods to aggregate (default: all four).")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config.experiment.results_dir)
    runs = _discover(results_dir, args.name, args.seeds, tuple(args.methods))
    if not runs:
        raise SystemExit("[aggregate] no completed runs found; nothing to do.")

    # Env-aware performance semantics: goal tasks succeed on termination;
    # survival tasks (CartPole) succeed by lasting to the horizon.
    family = make_family(config.env)
    success_on_termination = getattr(family.tasks[0], "success_on_termination", True)
    max_steps = int(config.env.params.get("max_steps", 0))

    report_dir = Path(args.reports_dir) / args.name
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    n_seeds = len(next(iter(runs.values())))
    print(f"[aggregate] {args.name}: methods={list(runs)} seeds={n_seeds}")

    # Headline seed-averaged figures (value-based).
    plot_retention_curves_ci(runs, figures_dir)
    plot_retention_bars_ci(runs, figures_dir)
    plot_average_performance_curve(runs, figures_dir)
    build_retention_table_ci(runs, figures_dir, tables_dir)

    # Seed-averaged forgetting matrix for every sequential method.
    for method in SEQUENTIAL:
        if method in runs:
            plot_forgetting_matrix_mean(
                runs[method], figures_dir / method,
                name="forgetting_matrix_mean")
    if "constrained" in runs:
        plot_forgetting_matrix_mean(runs["constrained"], figures_dir,
                                    name="forgetting_matrix_mean")

    # Concrete rollout performance (seed-averaged success rate / episode length).
    success, steps = _performance_stacks(config, runs, args.perf_episodes)
    build_performance_table_ci(success, steps, figures_dir, tables_dir,
                               success_on_termination=success_on_termination)
    # Interpretable headline bar chart: what the deployed policy actually does.
    if success_on_termination:
        plot_performance_bars_ci(
            {m: v * 100 for m, v in success.items()}, figures_dir,
            ylabel="Success rate (%)", title="Task success rate per task",
            filename="performance_bars", reference=100.0, reference_label="perfect")
    else:  # CartPole: balancing length is the graded headline metric.
        plot_performance_bars_ci(
            steps, figures_dir,
            ylabel="Mean balancing steps", title="Balancing length per task",
            filename="performance_bars",
            reference=max_steps or None,
            reference_label=f"horizon ({max_steps})" if max_steps else None)

    # Per-method diagnostics from seed 0 (dual dynamics + gaps are single-run).
    for method, dirs in runs.items():
        records = load_records(dirs[0])
        plot_duals(records, figures_dir / method)   # no-op without dual records
        plot_gaps(records, figures_dir / method)     # no-op without gap records
        with open(dirs[0] / "eval_matrix.json") as handle:
            plot_forgetting_matrix(json.load(handle), figures_dir / method)
        # Persist each seed's final eval matrix for provenance.
        for run_dir in dirs:
            seed = run_dir.name.split("seed")[-1]
            with open(run_dir / "eval_matrix.json") as src:
                (report_dir / f"eval_matrix_{method}_seed{seed}.json").write_text(src.read())

    print(f"[aggregate] report bundle written to {report_dir}")


if __name__ == "__main__":
    main()
