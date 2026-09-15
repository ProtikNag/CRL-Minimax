"""Large procedural gridworld family (no dense DP tensors, sampled rollouts).

TWO settings share this env; the config picks which:

  A. **Single-head, no-task-id** (policy kind ``mlp``, ``task_conditioned: false``,
     ``goal_in_obs: false``): the observation is only the agent's position and the
     network gets NO signal about which task it is on. A single policy
     pi(a | position) cannot point toward two different goals from the same cell,
     so perfect retention is infeasible by construction.
  B. **Task-incremental, per-task heads** (policy kind ``mlp_ac_multihead``, the
     PPO actor-critic backend; Lane A ``configs/biggrid_50task.yaml``): the task id
     selects a per-task actor+critic head, so the head resolves the goal and
     forgetting lives ONLY in the shared (deliberately narrow, e.g. [64]) trunk
     that all 50 heads contend for. Task id is available at train AND test (a
     legitimate task oracle for the task-incremental setting) -- state this when
     reporting. Difficulty comes from per-task DYNAMICS (slip, obstacle density)
     plus goal diversity, not from grid area.

Common design:
  * LARGE grid, computed procedurally -- no ``[S, A, S]`` tensor, so exact DP is
    not used (or feasible) at this scale. Values/gradients come only from sampled
    rollouts (REINFORCE or PPO), like MinAtar.
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


def _generate_blocked(size: int, density: float, goal: tuple[int, int],
                      seed: int) -> torch.Tensor:
    """Deterministic per-task obstacle layout as a [size, size] bool mask.

    ``density`` cells (fraction of the grid) are blocked, chosen uniformly but
    with the goal cell AND the goal's entire row and column reserved free. That
    reservation guarantees a Manhattan path from EVERY free cell to the goal
    (go along your column to the goal's row, then along the goal's row), so no
    task is ever unsolvable regardless of density. ``seed`` makes the layout a
    fixed property of the task (same across training seeds), so the dynamics --
    not RNG -- are what varies across tasks.
    """
    blocked = torch.zeros(size, size, dtype=torch.bool)
    if density <= 0.0:
        return blocked
    gr, gc = goal
    rng = np.random.default_rng(seed)
    reserved = np.zeros((size, size), dtype=bool)
    reserved[gr, :] = True          # goal row free
    reserved[:, gc] = True          # goal col free
    free_cells = np.argwhere(~reserved)
    n_block = min(int(round(density * size * size)), len(free_cells))
    if n_block <= 0:
        return blocked
    pick = rng.choice(len(free_cells), size=n_block, replace=False)
    for idx in pick:
        r, c = free_cells[idx]
        blocked[int(r), int(c)] = True
    return blocked


def _reachable_starts(blocked: torch.Tensor, goal: tuple[int, int]) -> torch.Tensor:
    """Free cells connected to the goal (excluding the goal itself), as a [M, 2]
    long tensor of (row, col). Starts are drawn ONLY from this component, so every
    episode is solvable regardless of obstacle density -- isolated free pockets are
    simply never used as spawn points (reserving the goal row/col alone does not
    guarantee global reachability, so we flood-fill instead of assuming it)."""
    size = blocked.shape[0]
    seen = torch.zeros_like(blocked)
    stack = [goal]
    seen[goal] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    seen[goal] = False  # spawn anywhere reachable EXCEPT on the goal
    return torch.nonzero(seen, as_tuple=False)


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

    def __init__(self, size, slip, goal, norm, gamma, max_steps, goal_in_obs,
                 blocked=None, starts=None):
        self.size = size
        self.slip = slip
        self.g_row, self.g_col = goal
        self.norm = norm
        self.gamma = gamma
        self.max_steps = max_steps
        self.goal_in_obs = goal_in_obs
        # blocked[r, c] True => moving into (r, c) is rejected (agent stays).
        self.blocked = (blocked.numpy() if isinstance(blocked, torch.Tensor)
                        else blocked)
        # [M, 2] valid spawn cells (goal-connected component); None => any cell.
        self.starts = (starts.numpy() if isinstance(starts, torch.Tensor)
                       else starts)
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
        if self.starts is not None:
            i = int(self.np_random.integers(len(self.starts)))
            self._row, self._col = int(self.starts[i, 0]), int(self.starts[i, 1])
        else:
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
        nr = min(max(self._row + dr, 0), self.size - 1)
        nc = min(max(self._col + dc, 0), self.size - 1)
        if self.blocked is None or not self.blocked[nr, nc]:
            self._row, self._col = nr, nc   # else: wall -> stay in place
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

    def __init__(self, spec, gamma, size, slip, goal, norm, max_steps, goal_in_obs,
                 blocked=None, threshold=float("inf")):
        super().__init__(spec, gamma)
        self.size = size
        self.slip = slip
        self.goal = goal
        self.norm = norm
        self.max_steps = max_steps
        self.goal_in_obs = goal_in_obs
        # Greedy-score target for the PPO local early-stop (inf => run to the cap).
        self.threshold = threshold
        # [size, size] bool mask of blocked cells (None => no obstacles).
        self.blocked = blocked
        # Valid spawn cells (goal-connected free component). Only computed when
        # obstacles exist; None => uniform sampling over non-goal cells (the
        # original obstacle-free behaviour, unchanged).
        self.starts = (_reachable_starts(blocked, goal)
                       if blocked is not None and bool(blocked.any()) else None)

    def make_env(self):
        return _BigGridEnv(self.size, self.slip, self.goal, self.norm,
                           self.gamma, self.max_steps, self.goal_in_obs,
                           blocked=self.blocked, starts=self.starts)

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        size = self.size
        g_row, g_col = self.goal
        tid = self.spec.task_id
        goal = self.goal if self.goal_in_obs else None
        n = num_episodes

        if self.starts is not None:
            # Spawn only in the goal-connected free component (every episode
            # solvable); isolated pockets are never used as starts.
            pick = torch.randint(0, self.starts.shape[0], (n,))
            row = self.starts[pick, 0].clone()
            col = self.starts[pick, 1].clone()
        else:
            row = torch.randint(0, size, (n,))
            col = torch.randint(0, size, (n,))
            on = (row == g_row) & (col == g_col)   # avoid spawning on the goal
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
            nrow = (row + _DROW[exec_a]).clamp(0, size - 1)
            ncol = (col + _DCOL[exec_a]).clamp(0, size - 1)
            if self.blocked is not None:
                # A move into a blocked cell is rejected: the agent stays put.
                open_move = ~self.blocked[nrow, ncol]
                row = torch.where(open_move, nrow, row)
                col = torch.where(open_move, ncol, col)
            else:
                row, col = nrow, ncol

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
        # Obstacle layouts are a fixed property of each task (varied dynamics, not
        # RNG): task i uses obstacle_seed + i. A per-task ``obstacle_density`` (or
        # the family default) sets how many cells are blocked; ``slip`` may also be
        # overridden per task. This is what makes the 50-task family HARD via
        # dynamics rather than grid area.
        obstacle_seed = int(params.get("obstacle_seed", 12345))
        default_density = float(params.get("obstacle_density", 0.0))
        default_threshold = float(params.get("threshold", float("inf")))
        if not tasks:
            raise ValueError("BigGridFamily needs a non-empty env.tasks list.")
        self.obs_dim = (4 if goal_in_obs else 2) * size
        self.num_actions = 4
        self.tasks = []
        for task_id, t in enumerate(tasks):
            gr, gc = t["goal"]
            t_slip = float(t.get("slip", slip))
            density = float(t.get("obstacle_density", default_density))
            threshold = float(t.get("threshold", default_threshold))
            blocked = _generate_blocked(size, density, (gr, gc),
                                        obstacle_seed + task_id)
            spec = TaskSpec(task_id, f"biggrid{size}-goal({gr},{gc})", t)
            self.tasks.append(BigGridTask(spec, gamma, size, t_slip, (gr, gc),
                                          norm, max_steps, goal_in_obs,
                                          blocked=blocked, threshold=threshold))
