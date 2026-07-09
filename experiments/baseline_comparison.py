"""Full baseline comparison: our method against the standard references.

Runs four training procedures on the same tasks, network, and estimator, so any
difference is due to the procedure alone:

    constrained    -- the min-max method (ours)
    finetune       -- ONE network trained on each task in order (the canonical
                      single-network sequential baseline; forgets old tasks)
    unconstrained  -- the two-policy method with the constraint switched off
                      (ablation; forgets the newest task)
    joint          -- one network trained on all tasks at once (upper bound)

The curated figure bundle and tables land under reports/<name>/ (tracked in
git). Raw runs stay under results/ (gitignored).

Usage:
    python -m experiments.baseline_comparison \
        --config configs/gridworld_nn_three_task.yaml --name nn_three_task [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from analysis.compare import (
    build_performance_table, build_retention_table, plot_retention_bars,
    plot_retention_curves,
)
from analysis.plots import (
    load_records, plot_duals, plot_forgetting_matrix, plot_gaps, plot_learning_curves,
)
from analysis.schematics import design_space_map, method_schematic
from crl.baselines import joint_multitask, sequential_finetune
from crl.buffers import BufferSet
from crl.config import Config, config_from_dict, load_config
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.evaluation import rollout_performance
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from experiments.run import run_from_config


def _run_custom(config: Config, train_fn, run_name: str) -> Path:
    """Build components and run a baseline training function; return run dir."""
    set_seed(config.experiment.seed)
    family = make_family(config.env)
    policy = make_policy(config.policy, family)
    estimator = make_estimator(config.estimator, buffer_set=BufferSet())
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())
    print(
        f"[run] name={run_name} seed={config.experiment.seed} "
        f"env={config.env.family}({len(family)} tasks) policy={config.policy.kind}"
    )
    try:
        train_fn(config, family, policy, estimator, logger)
    finally:
        logger.close()
    return logger.run_dir


def _single_run_figures(run_dir: Path, out_dir: Path) -> None:
    records = load_records(run_dir)
    plot_learning_curves(records, out_dir)
    plot_duals(records, out_dir)      # no-op for baselines without dual records
    plot_gaps(records, out_dir)       # no-op for baselines without gap records
    matrix_path = run_dir / "eval_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as handle:
            plot_forgetting_matrix(json.load(handle), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.seed is not None:
        base.experiment.seed = args.seed
    seed = base.experiment.seed
    raw = base.to_dict()
    results_dir = Path(raw["experiment"]["results_dir"])

    def variant(method: str):
        cfg = copy.deepcopy(raw)
        cfg["experiment"]["name"] = f"{args.name}_{method}"
        return cfg

    runs: dict[str, Path] = {}

    # Ours + the constraint-off ablation use the alternation trainer.
    print("\n=== constrained (ours) ===")
    run_from_config(config_from_dict(variant("constrained")))
    runs["constrained"] = results_dir / f"{args.name}_constrained_seed{seed}"

    print("\n=== unconstrained ablation (duals off) ===")
    ablation = variant("unconstrained")
    ablation["duals"]["lr"] = 0.0
    run_from_config(config_from_dict(ablation))
    runs["unconstrained"] = results_dir / f"{args.name}_unconstrained_seed{seed}"

    print("\n=== naive sequential fine-tuning (single network) ===")
    runs["finetune"] = _run_custom(
        config_from_dict(variant("finetune")), sequential_finetune,
        f"{args.name}_finetune_seed{seed}")

    print("\n=== joint multi-task (upper bound) ===")
    runs["joint"] = _run_custom(
        config_from_dict(variant("joint")), joint_multitask,
        f"{args.name}_joint_seed{seed}")

    # Figures.
    report_dir = Path(args.reports_dir) / args.name
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    # Per-run diagnostics.
    for method, run_dir in runs.items():
        _single_run_figures(run_dir, figures_dir / method)
        with open(run_dir / "eval_matrix.json") as src:
            (report_dir / f"eval_matrix_{method}.json").write_text(src.read())

    # Retention curves / bars over the SEQUENTIAL methods (joint has no task
    # order); tables include every method (joint is the upper-bound reference).
    sequential = {m: runs[m] for m in ("constrained", "finetune", "unconstrained")}
    plot_retention_curves(sequential, figures_dir)
    plot_retention_bars(sequential, figures_dir)
    build_retention_table(runs, figures_dir, tables_dir)

    # Concrete task performance from real rollouts of each final policy.
    family = make_family(base.env)
    perf_by_method = {}
    for method, run_dir in runs.items():
        policy = make_policy(base.policy, family)
        policy.load_state_dict(torch.load(run_dir / "final_policy.pt"))
        policy.eval()
        perf_by_method[method] = [
            rollout_performance(policy, task, num_episodes=300) for task in family.tasks
        ]
    build_performance_table(perf_by_method, figures_dir, tables_dir)

    design_space_map(figures_dir)
    method_schematic(figures_dir)

    print(f"\n[baselines] figure bundle written to {report_dir}")


if __name__ == "__main__":
    main()
