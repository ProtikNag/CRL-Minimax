"""Autonomous size-picker + config generator for the continual-maze study.

Runs a quick single-maze REINFORCE learnability check for a few candidate grid
sizes, picks the LARGEST size that a policy can actually solve (success >=
threshold), and writes the two run configs (constrained-local + unconstrained-
local) for a 20-maze continual study at that size. Used by the SLURM driver so
the whole pipeline is self-contained (no interactive size decision).
"""

from __future__ import annotations

import numpy as np
import torch

from crl.buffers import BufferSet
from crl.config import EnvConfig, EstimatorConfig, PolicyConfig
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.policies import make_policy
from crl.seeding import set_seed

CANDIDATES = [21, 15, 11]
THRESHOLD = 0.7
PROBE_STEPS = 1000


def _learn_success(size: int, steps: int = PROBE_STEPS) -> float:
    set_seed(0)
    env = EnvConfig(family="maze", params={
        "size": size, "braid": 0.08, "slip": 0.1, "gamma": 0.99,
        "goal_reward": 1.0, "shaping": 1.0, "view_k": 5,
        "max_steps": 4 * size, "wall_penalty_prob": 0.0}, tasks=[{"maze_seed": 0}])
    fam = make_family(env)
    task = fam.tasks[0]
    pol = make_policy(PolicyConfig(kind="mlp", hidden_sizes=[256, 256],
                                   task_conditioned=False), fam)
    est = make_estimator(EstimatorConfig(kind="monte_carlo", episodes_per_grad=64),
                         buffer_set=BufferSet())
    opt = torch.optim.Adam(pol.parameters(), lr=0.003)
    for _ in range(steps):
        obj, ent, _ = est.surrogate_objective(pol, task)
        loss = -(obj + 0.01 * ent)
        opt.zero_grad(); loss.backward(); opt.step()
    eps = task.vector_rollout(pol, 256)
    return float(np.mean([1.0 if e.terminated else 0.0 for e in eps]))


def _write_config(path: str, name: str, size: int, unconstrained: bool) -> None:
    tasks = "\n".join(f"    - {{ maze_seed: {i} }}" for i in range(20))
    variant = ("UNCONSTRAINED-LOCAL variant." if unconstrained
               else "Full theory (constrained local).")
    text = f"""# === 20-maze continual navigation ({size}x{size}), SAMPLED REINFORCE ===
# One distinct maze per task (walls, dead-ends, braided loops -> multiple paths).
# Start & goal randomized per episode (BFS-verified reachable). Observation =
# current cell + goal cell + local 5x5 wall patch (one shared head, NO task-id).
# Reward = BFS-shortest-path potential shaping + terminal goal reward + per-maze
# wall-hit penalty (0 for some mazes, a random negative for others); with
# discounting, shorter paths score higher. {variant}
experiment:
  name: {name}
  seed: 0
  results_dir: results
  device: cpu
  log_every: 100

env:
  family: maze
  params:
    size: {size}
    braid: 0.08
    slip: 0.1
    gamma: 0.99
    goal_reward: 1.0
    shaping: 1.0
    view_k: 5
    max_steps: {4 * size}
    wall_penalty_prob: 0.5
    wall_penalty_max: 0.3
  tasks:
{tasks}

policy:
  kind: mlp                    # ONE shared head (no per-task heads)
  hidden_sizes: [256, 256]
  task_conditioned: false      # NO task-id

estimator:
  kind: monte_carlo
  episodes_per_grad: 64
  episodes_per_eval: 64
  episodes_per_ref: 96
  baseline: batch_mean
  time_discount_weighting: true

duals:
  kind: projected_ascent
  lr: 1.0
  init: 0.0
  max_value: 100.0
  warm_start: true

trainer:
  cycles_per_task: 2
  local_steps: 600
  global_steps: 250
  task1_steps: 1200
  lr_local: 0.003
  lr_global: 0.003
  optimizer: adam
  eps: 0.05
  past_task_sampling: sample
  entropy_coef: 0.01
  eval_episodes: 64
  eval_all_tasks: true
  eval_probe_every: 100
  report_return: false         # discounted value = graded, prefers shorter paths
  local_unconstrained: {str(unconstrained).lower()}
"""
    with open(path, "w") as f:
        f.write(text)
    print(f"[pick] wrote {path}", flush=True)


def main() -> None:
    chosen = None
    for size in CANDIDATES:
        sr = _learn_success(size)
        print(f"[pick] maze {size}x{size}: single-maze success {sr:.2f}", flush=True)
        if sr >= THRESHOLD and chosen is None:
            chosen = size
    if chosen is None:
        chosen = CANDIDATES[-1]
        print(f"[pick] none reached {THRESHOLD}; falling back to {chosen}", flush=True)
    print(f"CHOSEN_SIZE {chosen}", flush=True)
    _write_config("configs/maze_20task.yaml", "maze_20task", chosen, False)
    _write_config("configs/maze_20task_localfree.yaml", "maze_20task_localfree",
                  chosen, True)


if __name__ == "__main__":
    main()
