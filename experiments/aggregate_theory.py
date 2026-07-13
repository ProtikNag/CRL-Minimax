"""Aggregate a 3-method min-max study (constrained-local, unconstrained-local,
fine-tune) into a report bundle, over N seeds.

The two "ours" variants come from two run-names (so they share every setting but
``local_unconstrained``); fine-tune comes from the constrained run-name:

    constrained  <- results/<constrained_name>_constrained_seed*   (full theory)
    localfree    <- results/<localfree_name>_constrained_seed*      (unconstrained local)
    finetune     <- results/<constrained_name>_finetune_seed*       (naive baseline)

The reported performance headline adapts to the env (from the config):
    report_return=True (MinAtar)  -> raw game-score table + bars
    success_on_termination=True   -> success-rate table + bars (gridworld)
    else                          -> balancing-steps table + bars

Writes reports/<name>/ with retention/learning/reward curves (mean±CI), the
seed-averaged forgetting matrix per method, the performance table/bars,
retention table, and per-method diagnostics (learning curves, duals, gaps).

Usage:
    # MinAtar (defaults):
    python -m experiments.aggregate_theory --seeds 0 1 2 3 4
    # GridWorld:
    python -m experiments.aggregate_theory --name gridworld_20task \
        --config configs/gridworld_20task.yaml \
        --constrained-name gridworld_20task --localfree-name gridworld_20task_localfree \
        --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.aggregate import (
    build_performance_table_ci, build_retention_table_ci, build_score_table_ci,
    plot_average_performance_curve, plot_forgetting_matrix_mean,
    plot_performance_bars_ci, plot_retention_bars_ci, plot_retention_curves_ci,
)
from analysis.plots import (
    load_records, plot_duals, plot_forgetting_matrix, plot_gaps,
    plot_learning_curves,
)
from crl.config import load_config
from crl.envs import make_family
from crl.evaluation import rollout_performance
from crl.policies import make_policy
from crl.seeding import set_seed
from experiments.aggregate_seeds import _performance_stacks, _raw_family


def _discover(results_dir: Path, sources: dict, seeds: list[int]) -> dict:
    runs = {}
    for key, (name, method) in sources.items():
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
    ap.add_argument("--name", default="minatar_theory", help="report bundle name")
    ap.add_argument("--config", default="configs/minatar_multihead.yaml")
    ap.add_argument("--constrained-name", default="minatar_multihead")
    ap.add_argument("--localfree-name", default="minatar_localfree")
    ap.add_argument("--perf-episodes", type=int, default=100)
    args = ap.parse_args()

    sources = {
        "constrained": (args.constrained_name, "constrained"),
        "localfree": (args.localfree_name, "constrained"),
        "finetune": (args.constrained_name, "finetune"),
    }
    results_dir = Path(args.results_dir)
    runs = _discover(results_dir, sources, args.seeds)
    if not runs:
        raise SystemExit("[aggregate-theory] no runs found.")
    n = min(len(v) for v in runs.values())
    config = load_config(args.config)
    report_return = getattr(config.trainer, "report_return", False)
    family = make_family(config.env)
    success_on_termination = getattr(family.tasks[0], "success_on_termination", True)
    metric_label = "score" if report_return else "value"
    print(f"[aggregate-theory] {args.name}: methods={list(runs)} seeds={n} "
          f"metric={metric_label}")

    report_dir = Path(args.reports_dir) / args.name
    figures = report_dir / "figures"
    tables = report_dir / "tables"

    # Seed-averaged headline figures (retention / learning / reward curves).
    plot_retention_curves_ci(runs, figures, metric_label=metric_label)
    plot_retention_bars_ci(runs, figures, metric_label=metric_label)
    plot_average_performance_curve(runs, figures, metric_label=metric_label)
    build_retention_table_ci(runs, figures, tables)
    for method, dirs in runs.items():
        plot_forgetting_matrix_mean(dirs, figures / method,
                                    name="forgetting_matrix_mean",
                                    metric_label=metric_label)

    # Concrete rollout performance of the final policy (env-adaptive headline).
    success, steps, returns = _performance_stacks(config, runs, args.perf_episodes)
    if report_return:  # MinAtar: raw game score
        game_names = [t.get("game", f"T{i+1}") for i, t in enumerate(config.env.tasks)]
        set_seed(123)
        raw_fam = _raw_family(config)
        rnd = make_policy(config.policy, raw_fam)
        rnd_scores = [rollout_performance(rnd, raw_fam.tasks[i], args.perf_episodes).mean_return
                      for i in range(len(raw_fam))]
        build_score_table_ci(returns, game_names, figures, tables, random_scores=rnd_scores)
        plot_performance_bars_ci(returns, figures, ylabel="Game score (raw mean return)",
                                 title="Actual game score per task", filename="performance_bars")
    elif success_on_termination:  # gridworld: goal-reaching success rate
        build_performance_table_ci(success, steps, figures, tables,
                                   success_on_termination=True)
        plot_performance_bars_ci({m: v * 100 for m, v in success.items()}, figures,
                                 ylabel="Success rate (%)",
                                 title="Task success rate per task",
                                 filename="performance_bars",
                                 reference=100.0, reference_label="perfect")
    else:  # survival tasks: balancing steps
        build_performance_table_ci(success, steps, figures, tables,
                                   success_on_termination=False)
        plot_performance_bars_ci(steps, figures, ylabel="Mean steps",
                                 title="Episode length per task", filename="performance_bars")

    # Per-method diagnostics from seed 0: learning/reward curves, duals, gaps, matrix.
    for method, dirs in runs.items():
        recs = load_records(dirs[0])
        plot_learning_curves(recs, figures / method, title_suffix=f" ({method})")
        plot_duals(recs, figures / method)
        plot_gaps(recs, figures / method)
        with open(dirs[0] / "eval_matrix.json") as h:
            plot_forgetting_matrix(json.load(h), figures / method)
        for run_dir in dirs:
            seed = run_dir.name.split("seed")[-1]
            (report_dir / f"eval_matrix_{method}_seed{seed}.json").write_text(
                (run_dir / "eval_matrix.json").read_text())

    print(f"[aggregate-theory] bundle -> {report_dir}")


if __name__ == "__main__":
    main()
