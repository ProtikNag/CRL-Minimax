"""Single-run entry point.

Usage:
    python -m experiments.run --config configs/gridworld_exact.yaml
    python -m experiments.run --config configs/minatar_multihead.yaml --seed 7
"""

from __future__ import annotations

import argparse

import torch

from crl.buffers import BufferSet
from crl.config import Config, load_config
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer


def resolve_device(name: str) -> torch.device:
    """Resolve a config device string; 'auto'/'cuda' fall back to CPU if no GPU."""
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_from_config(config: Config) -> list[list[float]]:
    """Build every component from config and train; returns the eval matrix."""
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    family = make_family(config.env)
    policy = make_policy(config.policy, family).to(device)
    run_name = f"{config.experiment.name}_seed{config.experiment.seed}"
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())

    if config.trainer.kind == "ppo":
        # PPO backend: same continual-learning framework, PPO as the optimizer.
        from crl.ppo_continual import PPOAlternationTrainer

        print(
            f"[run] name={run_name} seed={config.experiment.seed} "
            f"env={config.env.family}({len(family)} tasks) "
            f"policy={config.policy.kind} backend=ppo method={config.ppo.method} "
            f"eps={config.trainer.eps} device={device}"
        )
        trainer = PPOAlternationTrainer(config, family, policy, logger)
    else:
        estimator = make_estimator(config.estimator, buffer_set=BufferSet())
        print(
            f"[run] name={run_name} seed={config.experiment.seed} "
            f"env={config.env.family}({len(family)} tasks) "
            f"policy={config.policy.kind} estimator={config.estimator.kind} "
            f"eps={config.trainer.eps} device={device}"
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
