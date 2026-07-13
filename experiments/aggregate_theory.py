"""Aggregate the fresh MinAtar theory study (10 seeds) into the report bundle.

Three methods, drawn from two run-names (so the constrained-local and the
unconstrained-local variants share every other setting):

    constrained  <- results/minatar_multihead_constrained_seed*   (full theory)
    localfree    <- results/minatar_localfree_constrained_seed*    (unconstrained local)
    finetune     <- results/minatar_multihead_finetune_seed*       (naive baseline)

Writes reports/<name>/ with, in the reported game-score metric:
    figures/retention_curves_ci        per-task learning/reward curves (mean±CI)
    figures/retention_bars_ci          final per-task score, grouped bars ±CI
    figures/average_performance_curve  avg score over tasks-seen vs task count
    figures/performance_bars           actual raw game score per task ±CI
    figures/score_table                raw game-score table (+ random baseline)
    figures/retention_table            % retained table
    figures/<method>/                  per-method (seed 0): learning curves, duals,
                                       gaps, forgetting matrix
    tables/*.csv

Usage:
    python -m experiments.aggregate_theory --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.aggregate import (
    build_retention_table_ci, build_score_table_ci,
    plot_average_performance_curve, plot_forgetting_matrix_mean,
    plot_performance_bars_ci, plot_retention_bars_ci, plot_retention_curves_ci,
)
from analysis.plots import (
    load_records, plot_duals, plot_forgetting_matrix, plot_gaps,
    plot_learning_curves,
)
from crl.config import load_config
from crl.evaluation import rollout_performance
from crl.policies import make_policy
from crl.seeding import set_seed
from experiments.aggregate_seeds import _performance_stacks, _raw_family

# method key -> (run-name, method-name-in-dir)
SOURCES = {
    "constrained": ("minatar_multihead", "constrained"),
    "localfree": ("minatar_localfree", "constrained"),
    "finetune": ("minatar_multihead", "finetune"),
}


def _discover(results_dir: Path, seeds: list[int]) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for key, (name, method) in SOURCES.items():
        dirs = []
        for s in seeds:
            p = results_dir / f"{name}_{method}_seed{s}"
            if (p / "eval_matrix.json").exists():
                dirs.append(p)
            else:
                print(f"[aggregate-theory] missing: {p}")
        if dirs:
            runs[key] = dirs
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--name", default="minatar_theory")
    ap.add_argument("--perf-episodes", type=int, default=100)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs = _discover(results_dir, args.seeds)
    if not runs:
        raise SystemExit("[aggregate-theory] no runs found.")
    n = min(len(v) for v in runs.values())
    print(f"[aggregate-theory] methods={list(runs)} seeds={n}")

    config = load_config("configs/minatar_multihead.yaml")
    report_dir = Path(args.reports_dir) / args.name
    figures = report_dir / "figures"
    tables = report_dir / "tables"
    metric_label = "score"

    # Seed-averaged headline figures (game-score metric).
    plot_retention_curves_ci(runs, figures, metric_label=metric_label)
    plot_retention_bars_ci(runs, figures, metric_label=metric_label)
    plot_average_performance_curve(runs, figures, metric_label=metric_label)
    build_retention_table_ci(runs, figures, tables)
    for method, dirs in runs.items():
        plot_forgetting_matrix_mean(dirs, figures / method,
                                    name="forgetting_matrix_mean",
                                    metric_label=metric_label)

    # Actual raw game scores of the final policy (reward scaling stripped).
    _success, _steps, returns = _performance_stacks(config, runs, args.perf_episodes)
    game_names = [t["game"] for t in config.env.tasks]
    set_seed(123)
    raw_fam = _raw_family(config)
    rnd = make_policy(config.policy, raw_fam)
    rnd_scores = [rollout_performance(rnd, raw_fam.tasks[i], args.perf_episodes).mean_return
                  for i in range(len(raw_fam))]
    build_score_table_ci(returns, game_names, figures, tables, random_scores=rnd_scores)
    plot_performance_bars_ci(
        returns, figures, ylabel="Game score (raw mean return)",
        title="Actual game score per task", filename="performance_bars")

    # Per-method diagnostics from seed 0: learning/reward curves, duals, gaps, matrix.
    for method, dirs in runs.items():
        recs = load_records(dirs[0])
        plot_learning_curves(recs, figures / method, title_suffix=f" ({method})")
        plot_duals(recs, figures / method)
        plot_gaps(recs, figures / method)
        with open(dirs[0] / "eval_matrix.json") as h:
            plot_forgetting_matrix(json.load(h), figures / method)
        # provenance
        for run_dir in dirs:
            seed = run_dir.name.split("seed")[-1]
            (report_dir / f"eval_matrix_{method}_seed{seed}.json").write_text(
                (run_dir / "eval_matrix.json").read_text())

    print(f"[aggregate-theory] bundle -> {report_dir}")


if __name__ == "__main__":
    main()
