"""Single-run entry point.

Usage:
    python -m experiments.run --config configs/gridworld_exact.yaml
    python -m experiments.run --config configs/cartpole_family.yaml --seed 7
"""

from __future__ import annotations

import argparse

from crl.buffers import BufferSet
from crl.config import Config, load_config
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer


def run_from_config(config: Config) -> list[list[float]]:
    """Build every component from config and train; returns the eval matrix."""
    set_seed(config.experiment.seed)
    family = make_family(config.env)
    policy = make_policy(config.policy, family)
    estimator = make_estimator(config.estimator, buffer_set=BufferSet())
    run_name = f"{config.experiment.name}_seed{config.experiment.seed}"
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())

    # Run header (reproducibility convention: seed / env / model up front).
    print(
        f"[run] name={run_name} seed={config.experiment.seed} "
        f"env={config.env.family}({len(family)} tasks) "
        f"policy={config.policy.kind} estimator={config.estimator.kind} "
        f"eps={config.trainer.eps}"
    )

    trainer = AlternationTrainer(config, family, policy, estimator, logger)
    try:
        matrix = trainer.run()
    finally:
        logger.close()
    print(f"[run] done; results in {logger.run_dir}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument(
        "--results-dir", default=None, help="Override results directory."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.results_dir is not None:
        config.experiment.results_dir = args.results_dir
    run_from_config(config)


if __name__ == "__main__":
    main()
