"""Entry point for the Double-DQN realization of the min-max method.

Two modes:
    continual  -- run the constrained local/global DDQN trainer over the task
                  sequence (crl/dqn.py). Writes results/<name>_dqn_seed<seed>/.
    experts    -- train a dedicated single-game DDQN agent per task (same net
                  architecture, one game each) to serve as the empirical
                  "highest possible score" reference ceiling per game (MinAtar
                  has no fixed max like Pong's +21). Writes one run dir per game.

Usage:
    python -m experiments.run_dqn --config configs/minatar_dqn.yaml --mode continual
    python -m experiments.run_dqn --config configs/minatar_dqn.yaml --mode experts
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn

from crl.config import load_config
from crl.dqn import (
    DQNContinualTrainer, EnvPool, ReplayBuffer, _epsilon, _sync, ddqn_loss,
    greedy_value,
)
from crl.envs import make_family
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from experiments.run import resolve_device


def run_continual(config, seed: int) -> None:
    set_seed(seed)
    device = resolve_device(config.experiment.device)
    family = make_family(config.env)
    qnet = make_policy(config.policy, family).to(device)
    run_name = f"{config.experiment.name}_seed{seed}"
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())
    print(f"[dqn] {run_name} device={device} tasks={len(family)} "
          f"policy={config.policy.kind}")
    trainer = DQNContinualTrainer(config, family, qnet, logger, device)
    try:
        trainer.run()
    finally:
        logger.close()
    print(f"[dqn] done; results in {logger.run_dir}")


def train_expert(qnet, task, cfg, gamma, obs_shape, device, logger) -> float:
    """Dedicated single-game DDQN; returns the final raw game score."""
    buf = ReplayBuffer(cfg.buffer_capacity, obs_shape)
    pool = EnvPool(task, cfg.num_envs)
    target = copy.deepcopy(qnet)
    opt = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)
    tid = task.spec.task_id
    while len(buf) < cfg.warmup_transitions:
        pool.collect(qnet, 1.0, cfg.num_envs * 4, buf, device)
    for step in range(cfg.task1_steps):
        eps = _epsilon(step, cfg)
        pool.collect(qnet, eps, cfg.collect_per_step, buf, device)
        batch = buf.sample(cfg.batch_size, device)
        loss = ddqn_loss(qnet, target, batch, tid, gamma)
        opt.zero_grad(); loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(qnet.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.target_sync_every == 0:
            _sync(target, qnet)
        if step % cfg.log_every == 0:
            v, score = greedy_value(qnet, task, cfg.value_episodes, gamma, device)
            logger.log({"phase": "expert", "task": tid + 1, "step": step,
                        "loss": float(loss), "V": v, "score": score})
            print(f"[expert t{tid+1} {task._game}] step={step:6d} "
                  f"loss={float(loss):.4f} score={score:.2f}")
    v, score = greedy_value(qnet, task, cfg.eval_episodes, gamma, device)
    return score


def run_finetune(config, seed: int) -> None:
    """Naive sequential DDQN baseline: one shared net trained task-by-task with
    NO constraint (no global/local split). Shows catastrophic forgetting."""
    set_seed(seed)
    device = resolve_device(config.experiment.device)
    family = make_family(config.env)
    qnet = make_policy(config.policy, family).to(device)
    gamma = family.tasks[0].gamma
    run_name = f"{config.experiment.name}_finetune_seed{seed}"
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())
    print(f"[dqn-finetune] {run_name} device={device} tasks={len(family)}")
    eval_matrix = []
    try:
        for k, task in enumerate(family.tasks):
            print(f"\n=== finetune task {k+1}/{len(family)}: {task._game} ===")
            # train the SHARED net on this task alone (steps = task1 budget).
            train_expert(qnet, task, config.dqn, gamma, family.obs_shape,
                         device, logger)
            row = [greedy_value(qnet, family.tasks[i], config.dqn.eval_episodes,
                                gamma, device)[1] for i in range(k + 1)]
            eval_matrix.append(row)
            logger.log({"phase": "eval", "task": k + 1, "values": row})
            print(f"[eval t{k+1}] {row}")
        logger.save_json("eval_matrix.json", eval_matrix)
        torch.save(qnet.state_dict(), logger.run_dir / "final_policy.pt")
    finally:
        logger.close()
    print(f"[dqn-finetune] done; results in {logger.run_dir}")


def run_experts(config, seed: int) -> None:
    set_seed(seed)
    device = resolve_device(config.experiment.device)
    family = make_family(config.env)
    ceilings = {}
    for i, task in enumerate(family.tasks):
        print(f"\n=== expert {i+1}/{len(family)}: {task._game} ===")
        qnet = make_policy(config.policy, family).to(device)
        run_name = f"{config.experiment.name}_expert_{task._game}_seed{seed}"
        logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict())
        try:
            score = train_expert(qnet, task, config.dqn, family.tasks[0].gamma,
                                 family.obs_shape, device, logger)
            logger.save_json("expert_score.json", {"game": task._game, "score": score})
            torch.save(qnet.state_dict(), logger.run_dir / "final_policy.pt")
        finally:
            logger.close()
        ceilings[task._game] = score
        print(f"[expert] {task._game} ceiling score = {score:.2f}")
    print(f"\n[experts] per-game ceilings: {ceilings}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["continual", "experts", "finetune"],
                        default="continual")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.results_dir is not None:
        config.experiment.results_dir = args.results_dir
    seed = config.experiment.seed
    if args.mode == "continual":
        run_continual(config, seed)
    elif args.mode == "finetune":
        run_finetune(config, seed)
    else:
        run_experts(config, seed)


if __name__ == "__main__":
    main()
