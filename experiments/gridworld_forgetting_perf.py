"""Rerun the gridworld 20-task study and record a SUCCESS-RATE forgetting matrix.

The stored ``eval_matrix.json`` holds discounted VALUE. GridWorld is tabular +
exact + deterministic (fixed seed), so rerunning reproduces the identical run;
here we additionally evaluate, after finishing each task k, the goal-reaching
SUCCESS RATE (fraction of episodes that terminate at the goal) on every task i,
building an actual-performance forgetting matrix ``perf_matrix`` in [0,1].

    python -m experiments.gridworld_forgetting_perf --config configs/gridworld_20task.yaml \
        --name gridworld_20task_constrained --seed 0 [--localfree] [--finetune]

Saves ``results/<name>_seed<seed>/perf_matrix.json``.
"""

from __future__ import annotations

import argparse
import json

import torch

from crl.buffers import BufferSet
from crl.config import load_config
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer


@torch.no_grad()
def _success_rate(policy, task, num_episodes: int) -> float:
    """Fraction of episodes that terminate (reach the goal) under ``policy``."""
    episodes = task.vector_rollout(policy, num_episodes)
    if not episodes:
        return 0.0
    return sum(1 for e in episodes if e.terminated) / len(episodes)


class PerfTrainer(AlternationTrainer):
    """AlternationTrainer that also records a success-rate forgetting matrix."""

    def __init__(self, *a, perf_episodes: int = 200, **k):
        super().__init__(*a, **k)
        self.perf_matrix: list[list[float]] = []
        self._perf_episodes = perf_episodes

    def _perf_row(self, k: int) -> list[float]:
        last = len(self.family) if self.cfg.eval_all_tasks else k
        return [_success_rate(self.global_policy, self.family.tasks[i],
                              self._perf_episodes) for i in range(last)]

    def run(self):
        # Mirror the base loop, snapshotting the SUCCESS matrix after each task.
        self._train_first_task()
        self.eval_matrix.append(self._evaluate_row(1))
        self.perf_matrix.append(self._perf_row(1))
        for k in range(2, len(self.family) + 1):
            for cycle in range(self.cfg.cycles_per_task):
                frozen_local = self._local_phase(k, cycle)
                self._global_phase(k, cycle, frozen_local)
                self._log_gaps(k, cycle, frozen_local)
            self.eval_matrix.append(self._evaluate_row(k))
            self.perf_matrix.append(self._perf_row(k))
            print(f"[perf] task {k}: diag success="
                  f"{self.perf_matrix[-1][k-1]:.2f}")
        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        self.logger.save_json("perf_matrix.json", self.perf_matrix)
        return self.perf_matrix


def run_finetune_perf(cfg, family, policy, estimator, logger, perf_episodes):
    """Naive sequential fine-tuning (matches crl.baselines.sequential_finetune),
    snapshotting the success-rate row after each task."""
    from crl.trainer import _make_optimizer
    tcfg = cfg.trainer
    later = tcfg.cycles_per_task * (tcfg.local_steps + tcfg.global_steps)
    perf: list[list[float]] = []
    last = len(family)
    for k in range(1, len(family) + 1):
        task = family.tasks[k - 1]
        steps = tcfg.task1_steps if k == 1 else later
        optimizer = _make_optimizer(tcfg.optimizer, policy.parameters(), tcfg.lr_global)
        for _ in range(steps):
            obj, ent, _ = estimator.surrogate_objective(policy, task)
            loss = -(obj + tcfg.entropy_coef * ent)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        perf.append([_success_rate(policy, family.tasks[i], perf_episodes)
                     for i in range(last)])
        print(f"[perf] finetune task {k}: diag success={perf[-1][k-1]:.2f}")
    logger.save_json("perf_matrix.json", perf)
    return perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perf-episodes", type=int, default=200)
    ap.add_argument("--localfree", action="store_true",
                    help="local_unconstrained variant (matches *_localfree runs)")
    ap.add_argument("--finetune", action="store_true",
                    help="naive sequential fine-tuning baseline")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.experiment.seed = args.seed
    cfg.experiment.name = args.name
    if args.localfree:
        cfg.trainer.local_unconstrained = True

    set_seed(cfg.experiment.seed)
    family = make_family(cfg.env)
    policy = make_policy(cfg.policy, family)
    estimator = make_estimator(cfg.estimator, buffer_set=BufferSet())
    run_name = f"{cfg.experiment.name}_seed{cfg.experiment.seed}"
    logger = RunLogger(cfg.experiment.results_dir, run_name, cfg.to_dict())
    print(f"[perf-rerun] {run_name} localfree={args.localfree} finetune={args.finetune}")
    try:
        if args.finetune:
            run_finetune_perf(cfg, family, policy, estimator, logger, args.perf_episodes)
        else:
            PerfTrainer(cfg, family, policy, estimator, logger,
                        perf_episodes=args.perf_episodes).run()
    finally:
        logger.close()
    print(f"[perf-rerun] done -> {logger.run_dir}/perf_matrix.json")


if __name__ == "__main__":
    main()
