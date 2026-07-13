"""Large procedural gridworld family (no dense DP tensors, sampled REINFORCE).

Design goals (progressively hardened, no crutches):
  * LARGE grid, computed procedurally -- no ``[S, A, S]`` tensor, so exact DP is
    not used (or feasible) at this scale. Values/gradients come only from
    sampled REINFORCE rollouts, like MinAtar.
  * ONE shared action head (policy kind ``mlp``) -- no per-task heads.
  * NO task-id given to the policy (``task_conditioned: false``) and NO goal in
    the observation by default: the observation is only the agent's position
    (factored one-hot ``one_hot(row) ++ one_hot(col)``, length ``2*size``). The
    network therefore gets NO signal about which task it is on. A single policy
    pi(a | position) cannot point toward two different goals from the same cell,
    so perfect retention is infeasible: the constraint must find the best
    *single* policy that stays broadly competent across all goals, while naive
    fine-tuning collapses onto the latest goal.
  * GRADED proximity reward (not 0/1 success): the episode return is the
    proximity of the closest approach to the goal, prox(d) = max(0, 1 - d/norm)
    in [0, 1], reaching the goal gives 1. So getting into the goal's periphery
    earns partial, graded credit -- essential here, since a compromise policy
    would score 0 under a binary metric. Reward is the increase in the running
    maximum proximity (dense, so REINFORCE gets a gradient from any start, and
    the undiscounted return telescopes to prox(closest approach)).

Set ``goal_in_obs: true`` to append the goal's factored one-hot to the
observation (a goal-conditioned variant where retention IS achievable); default
is False -- the hard, no-task-signal setting above.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from crl.buffers import Trajectory
from crl.envs.base import Task, TaskFamily, TaskSpec

_DROW = torch.tensor([-1, 1, 0, 0])
_DCOL = torch.tensor([0, 0, -1, 1])
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pos_obs(row: torch.Tensor, col: torch.Tensor, size: int,
             goal: tuple[int, int] | None) -> torch.Tensor:
    """[N] row,col -> factored one-hot [N, 2*size] (++ goal one-hot if given)."""
    n = row.shape[0]
    idx = torch.arange(n)
    blocks = [torch.zeros(n, size), torch.zeros(n, size)]
    blocks[0][idx, row] = 1.0
    blocks[1][idx, col] = 1.0
    if goal is not None:
        gr = torch.zeros(n, size); gc = torch.zeros(n, size)
        gr[:, goal[0]] = 1.0; gc[:, goal[1]] = 1.0
        blocks += [gr, gc]
    return torch.cat(blocks, dim=1)


class _BigGridEnv(gym.Env):
    """Single-episode procedural gridworld (used by the reporting rollout)."""

    def __init__(self, size, slip, goal, norm, gamma, max_steps, goal_in_obs):
        self.size = size
        self.slip = slip
        self.g_row, self.g_col = goal
        self.norm = norm
        self.gamma = gamma
        self.max_steps = max_steps
        self.goal_in_obs = goal_in_obs
        obs_dim = (4 if goal_in_obs else 2) * size
        self.observation_space = gym.spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def _prox(self, r, c):
        d = abs(r - self.g_row) + abs(c - self.g_col)
        return max(0.0, 1.0 - d / self.norm)

    def _obs(self):
        o = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        o[self._row] = 1.0
        o[self.size + self._col] = 1.0
        if self.goal_in_obs:
            o[2 * self.size + self.g_row] = 1.0
            o[3 * self.size + self.g_col] = 1.0
        return o

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self._row = int(self.np_random.integers(self.size))
            self._col = int(self.np_random.integers(self.size))
            if (self._row, self._col) != (self.g_row, self.g_col):
                break
        self._steps = 0
        self._best = self._prox(self._row, self._col)
        # return the starting proximity as the first reward (baseline offset)
        self._pending0 = self._best
        return self._obs(), {}

    def step(self, action):
        if self.np_random.random() < self.slip:
            action = int(self.np_random.integers(4))
        dr, dc = _MOVES[action]
        self._row = min(max(self._row + dr, 0), self.size - 1)
        self._col = min(max(self._col + dc, 0), self.size - 1)
        self._steps += 1
        prox = self._prox(self._row, self._col)
        reward = max(0.0, prox - self._best)
        if self._steps == 1:
            reward += self._pending0  # emit the starting proximity once
        self._best = max(self._best, prox)
        terminated = (self._row, self._col) == (self.g_row, self.g_col)
        truncated = self._steps >= self.max_steps
        return self._obs(), float(reward), terminated, bool(truncated and not terminated), {}


class BigGridTask(Task):
    """One large-grid goal task; graded proximity return in [0, 1]."""

    success_on_termination = True  # reaching the goal terminates (prox = 1)

    def __init__(self, spec, gamma, size, slip, goal, norm, max_steps, goal_in_obs):
        super().__init__(spec, gamma)
        self.size = size
        self.slip = slip
        self.goal = goal
        self.norm = norm
        self.max_steps = max_steps
        self.goal_in_obs = goal_in_obs

    def make_env(self):
        return _BigGridEnv(self.size, self.slip, self.goal, self.norm,
                           self.gamma, self.max_steps, self.goal_in_obs)

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        size = self.size
        g_row, g_col = self.goal
        tid = self.spec.task_id
        goal = self.goal if self.goal_in_obs else None
        n = num_episodes

        row = torch.randint(0, size, (n,))
        col = torch.randint(0, size, (n,))
        on = (row == g_row) & (col == g_col)
        row[on] = (row[on] + 1) % size

        def prox(r, c):
            d = (r - g_row).abs() + (c - g_col).abs()
            return (1.0 - d.float() / self.norm).clamp(min=0.0)

        best = prox(row, col)  # running-max proximity; starting proximity counts
        alive = torch.ones(n, dtype=torch.bool)
        reached = torch.zeros(n, dtype=torch.bool)
        lengths = torch.zeros(n, dtype=torch.long)
        obs_hist, act_hist, rew_hist, logp_hist = [], [], [], []
        first = True

        for _ in range(self.max_steps):
            obs = _pos_obs(row, col, size, goal)
            dist = policy.dist(obs, tid)
            action = dist.sample()
            logp = dist.log_prob(action)

            slip_mask = torch.rand(n) < self.slip
            exec_a = torch.where(slip_mask, torch.randint(0, 4, (n,)), action)
            row = (row + _DROW[exec_a]).clamp(0, size - 1)
            col = (col + _DCOL[exec_a]).clamp(0, size - 1)

            p = prox(row, col)
            reward = (p - best).clamp(min=0.0)
            if first:
                reward = reward + best  # emit starting proximity once (t=0)
                first = False
            best = torch.maximum(best, p)

            obs_hist.append(obs)
            act_hist.append(action)
            rew_hist.append(reward)
            logp_hist.append(logp)

            lengths = torch.where(alive, lengths + 1, lengths)
            hit = alive & (row == g_row) & (col == g_col)
            reached = reached | hit
            alive = alive & ~hit
            if not bool(alive.any()):
                break

        obs_s = torch.stack(obs_hist); act_s = torch.stack(act_hist)
        rew_s = torch.stack(rew_hist); logp_s = torch.stack(logp_hist)
        episodes = []
        for i in range(n):
            L = int(lengths[i])
            episodes.append(Trajectory(
                obs=obs_s[:L, i], actions=act_s[:L, i], rewards=rew_s[:L, i],
                behavior_logps=logp_s[:L, i], terminated=bool(reached[i])))
        return episodes


class BigGridFamily(TaskFamily):
    """Large procedural gridworld family (sampled-only, graded proximity).

    Family params (``env.params``):
        size: grid side length (default 50)
        slip: prob the executed action is uniformly random (default 0.1)
        gamma: discount (default 0.99)
        prox_norm: distance normalizer for prox(d)=max(0,1-d/norm); default
                   2*(size-1) (graded over the whole grid)
        max_steps: episode horizon (default 200)
        goal_in_obs: append the goal's factored one-hot to the observation
                     (goal-conditioned variant; default False)

    Per-task params: goal: [row, col] (required).
    """

    is_tabular = False

    def __init__(self, params: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        size = int(params.get("size", 50))
        slip = float(params.get("slip", 0.1))
        gamma = float(params.get("gamma", 0.99))
        norm = float(params.get("prox_norm", 2 * (size - 1)))
        max_steps = int(params.get("max_steps", 200))
        goal_in_obs = bool(params.get("goal_in_obs", False))
        if not tasks:
            raise ValueError("BigGridFamily needs a non-empty env.tasks list.")
        self.obs_dim = (4 if goal_in_obs else 2) * size
        self.num_actions = 4
        self.tasks = []
        for task_id, t in enumerate(tasks):
            gr, gc = t["goal"]
            spec = TaskSpec(task_id, f"biggrid{size}-goal({gr},{gc})", t)
            self.tasks.append(BigGridTask(spec, gamma, size, slip, (gr, gc),
                                          norm, max_steps, goal_in_obs))
