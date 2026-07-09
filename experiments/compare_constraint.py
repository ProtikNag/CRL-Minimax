"""Run the constrained method against the unconstrained baseline and build the
full figure bundle for one experiment.

The baseline is the identical config with the duals disabled (``duals.lr = 0``),
which switches off the min-max constraint and nothing else. Raw runs land under
``results/`` (gitignored); the curated, committable figure bundle and tables
land under ``reports/<name>/`` so they can be pushed and pulled from HPC.

Usage:
    python -m experiments.compare_constraint \
        --config configs/gridworld_nn_three_task.yaml --name nn_three_task [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from analysis.compare import build_comparison, build_performance_table
from analysis.plots import (
    load_records, plot_duals, plot_forgetting_matrix, plot_gaps, plot_learning_curves,
)
from analysis.schematics import design_space_map, method_schematic
from crl.config import config_from_dict, load_config
from crl.envs import make_family
from crl.evaluation import rollout_performance
from crl.policies import make_policy
from experiments.run import run_from_config


def _single_run_figures(run_dir: Path, out_dir: Path) -> None:
    records = load_records(run_dir)
    plot_learning_curves(records, out_dir)
    plot_duals(records, out_dir)
    plot_gaps(records, out_dir)
    matrix_path = run_dir / "eval_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as handle:
            plot_forgetting_matrix(json.load(handle), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True, help="Experiment name (reports/<name>).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.seed is not None:
        base.experiment.seed = args.seed
    seed = base.experiment.seed
    raw = base.to_dict()

    # Constrained (as configured) and unconstrained (duals off) variants.
    variants = {}
    for method, dual_lr in (("constrained", raw["duals"]["lr"]), ("unconstrained", 0.0)):
        cfg_raw = copy.deepcopy(raw)
        cfg_raw["duals"]["lr"] = dual_lr
        cfg_raw["experiment"]["name"] = f"{args.name}_{method}"
        print(f"\n=== running {method} (duals.lr={dual_lr}) ===")
        run_from_config(config_from_dict(cfg_raw))
        variants[method] = Path(cfg_raw["experiment"]["results_dir"]) / \
            f"{args.name}_{method}_seed{seed}"

    report_dir = Path(args.reports_dir) / args.name
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    # Per-run diagnostics into method-tagged subfolders.
    for method, run_dir in variants.items():
        _single_run_figures(run_dir, figures_dir / method)
        # Copy the eval matrix into the committable report.
        with open(run_dir / "eval_matrix.json") as src:
            (report_dir / f"eval_matrix_{method}.json").write_text(src.read())

    # Concrete task performance (success rate / steps) from real rollouts of the
    # final GLOBAL policy -- the interpretable metric, not the optimized value.
    family = make_family(base.env)
    perf_by_method = {}
    for method, run_dir in variants.items():
        policy = make_policy(base.policy, family)
        policy.load_state_dict(torch.load(run_dir / "final_policy.pt"))
        policy.eval()
        perf_by_method[method] = [
            rollout_performance(policy, task, num_episodes=300) for task in family.tasks
        ]

    # Cross-method comparison figures + retention/performance tables, schematics.
    build_comparison(variants, figures_dir, tables_dir)
    build_performance_table(perf_by_method, figures_dir, tables_dir)
    design_space_map(figures_dir)
    method_schematic(figures_dir)

    print(f"\n[compare] figure bundle written to {report_dir}")
    print(f"          PNG:  {figures_dir}/png/    SVG: {figures_dir}/svg/")
    print(f"          per-method diagnostics under {figures_dir}/<method>/")


if __name__ == "__main__":
    main()
