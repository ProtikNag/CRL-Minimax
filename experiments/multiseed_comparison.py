"""Run all four methods for a SINGLE seed (one point of a multi-seed study).

This is the per-seed unit of the headline experiment. It runs, on identical
tasks / network / estimator so only the procedure differs:

    constrained    -- the min-max method (ours)
    unconstrained  -- the two-policy method with the constraint switched off
    finetune       -- one network trained on each task in order (forgets old)
    joint          -- one network trained on all tasks at once (upper bound)

It writes raw run directories under results/ and produces NO figures; the
seed-averaged figure bundle is built afterwards by experiments/aggregate_seeds.py
once every seed has run. Splitting run from aggregate lets each seed map onto a
SLURM array index (--seed) and run in parallel.

Usage:
    python -m experiments.multiseed_comparison \
        --config configs/gridworld_tentask_sampled.yaml --name gridworld_tentask --seed 0
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from crl.baselines import joint_multitask, sequential_finetune
from crl.buffers import BufferSet
from crl.config import Config, config_from_dict, load_config
from crl.envs import make_family
from crl.estimators import make_estimator
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
    print(f"[run] {run_name} seed={config.experiment.seed} "
          f"env={config.env.family}({len(family)} tasks)")
    try:
        train_fn(config, family, policy, estimator, logger)
    finally:
        logger.close()
    return logger.run_dir


ALL_METHODS = ("constrained", "unconstrained", "finetune", "joint")


def run_one_seed(config_path: str, name: str, seed: int,
                 methods: tuple[str, ...] = ALL_METHODS) -> dict[str, Path]:
    """Run the selected methods at ``seed``; return {method: run_dir}."""
    base = load_config(config_path)
    base.experiment.seed = seed
    raw = base.to_dict()
    results_dir = Path(raw["experiment"]["results_dir"])

    def variant(method: str):
        cfg = copy.deepcopy(raw)
        cfg["experiment"]["name"] = f"{name}_{method}"
        cfg["experiment"]["seed"] = seed
        return cfg

    runs: dict[str, Path] = {}

    if "constrained" in methods:
        print(f"\n=== [seed {seed}] constrained (ours) ===")
        run_from_config(config_from_dict(variant("constrained")))
        runs["constrained"] = results_dir / f"{name}_constrained_seed{seed}"

    if "unconstrained" in methods:
        print(f"\n=== [seed {seed}] unconstrained ablation (duals off) ===")
        ablation = variant("unconstrained")
        ablation["duals"]["lr"] = 0.0
        run_from_config(config_from_dict(ablation))
        runs["unconstrained"] = results_dir / f"{name}_unconstrained_seed{seed}"

    if "finetune" in methods:
        print(f"\n=== [seed {seed}] naive sequential fine-tuning ===")
        runs["finetune"] = _run_custom(
            config_from_dict(variant("finetune")), sequential_finetune,
            f"{name}_finetune_seed{seed}")

    if "joint" in methods:
        print(f"\n=== [seed {seed}] joint multi-task (upper bound) ===")
        runs["joint"] = _run_custom(
            config_from_dict(variant("joint")), joint_multitask,
            f"{name}_joint_seed{seed}")

    print(f"\n[multiseed] seed {seed} complete: {[str(p) for p in runs.values()]}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS),
                        choices=ALL_METHODS,
                        help="Subset of methods to run (default: all four).")
    args = parser.parse_args()
    run_one_seed(args.config, args.name, args.seed, tuple(args.methods))


if __name__ == "__main__":
    main()
