"""Parametric CartPole family (development tier, sampled estimators only).

Each task overrides physics parameters of ``CartPole-v1`` (pole length, pole
mass, gravity, force magnitude). Dynamics change across tasks while state and
action spaces stay fixed, giving a cheap non-tabular testbed for the sampled
estimators and the alternation loop.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import torch

from crl.buffers import Trajectory
from crl.envs.base import Task, TaskFamily, TaskSpec

# Physics attributes that may be overridden per task.
_PHYSICS_KEYS = ("length", "masspole", "masscart", "gravity", "force_mag")

# CartPole-v1 constants (gymnasium classic_control defaults).
_TAU = 0.02
_THETA_THRESHOLD = 12 * 2 * math.pi / 360
_X_THRESHOLD = 2.4
_DEFAULTS = {"gravity": 9.8, "masscart": 1.0, "masspole": 0.1,
             "length": 0.5, "force_mag": 10.0}


class CartPoleTask(Task):
    """CartPole-v1 with per-task physics overrides.

    Termination means the pole fell (or the cart left bounds) -- a *failure* --
    so success is surviving to the horizon, not terminating.
    """

    success_on_termination = False

    def __init__(self, spec: TaskSpec, gamma: float, max_steps: int) -> None:
        super().__init__(spec, gamma)
        self._max_steps = max_steps
        for key in spec.params:
            if key not in _PHYSICS_KEYS:
                raise KeyError(
                    f"Unknown CartPole physics key '{key}'; allowed: {_PHYSICS_KEYS}"
                )
        self._phys = {**_DEFAULTS, **{k: float(v) for k, v in spec.params.items()}}

    def make_env(self) -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=self._max_steps)
        core = env.unwrapped
        for key, value in self.spec.params.items():
            setattr(core, key, float(value))
        # CartPoleEnv derives these in __init__; recompute after overrides.
        core.total_mass = core.masspole + core.masscart
        core.polemass_length = core.masspole * core.length
        return env

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        """Collect ``num_episodes`` episodes in lockstep with batched torch
        dynamics that replicate gym's Euler CartPole step-for-step.

        Stepping all N episodes with one batched policy forward pass per
        timestep (instead of the per-episode gym loop) is the CartPole analogue
        of the gridworld fast path: essential here because a well-balancing
        policy produces long episodes (up to the horizon). State is kept in
        float64 to match gym exactly; the policy sees the float32 observation.
        """
        p = self._phys
        total_mass = p["masspole"] + p["masscart"]
        polemass_length = p["masspole"] * p["length"]
        gravity, length, masspole = p["gravity"], p["length"], p["masspole"]
        force_mag = p["force_mag"]
        task_id = self.spec.task_id

        # Reset: uniform(-0.05, 0.05) on all four state dims (gym default).
        state = (torch.rand(num_episodes, 4, dtype=torch.float64) * 0.1) - 0.05
        alive = torch.ones(num_episodes, dtype=torch.bool)
        fell = torch.zeros(num_episodes, dtype=torch.bool)
        lengths = torch.zeros(num_episodes, dtype=torch.long)

        obs_hist: list[torch.Tensor] = []
        act_hist: list[torch.Tensor] = []
        rew_hist: list[torch.Tensor] = []
        logp_hist: list[torch.Tensor] = []

        for _ in range(self._max_steps):
            obs = state.to(torch.float32)  # gym returns float32(state)
            dist = policy.dist(obs, task_id)
            action = dist.sample()  # [N] in {0, 1}
            logp = dist.log_prob(action)

            x, x_dot, theta, theta_dot = state.unbind(dim=1)
            force = torch.where(action == 1, force_mag, -force_mag).to(torch.float64)
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)
            temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
            xacc = temp - polemass_length * thetaacc * costheta / total_mass
            x = x + _TAU * x_dot
            x_dot = x_dot + _TAU * xacc
            theta = theta + _TAU * theta_dot
            theta_dot = theta_dot + _TAU * thetaacc
            state = torch.stack([x, x_dot, theta, theta_dot], dim=1)

            # Reward is +1 for every step taken, including the terminal one.
            obs_hist.append(obs)
            act_hist.append(action)
            rew_hist.append(torch.ones(num_episodes, dtype=torch.float32))
            logp_hist.append(logp)

            failed = (x < -_X_THRESHOLD) | (x > _X_THRESHOLD) | \
                     (theta < -_THETA_THRESHOLD) | (theta > _THETA_THRESHOLD)
            lengths = torch.where(alive, lengths + 1, lengths)
            just_failed = alive & failed
            fell = fell | just_failed
            alive = alive & ~failed
            if not bool(alive.any()):
                break

        obs_stack = torch.stack(obs_hist)
        act_stack = torch.stack(act_hist)
        rew_stack = torch.stack(rew_hist)
        logp_stack = torch.stack(logp_hist)

        episodes: list[Trajectory] = []
        for n in range(num_episodes):
            length_n = int(lengths[n])
            episodes.append(
                Trajectory(
                    obs=obs_stack[:length_n, n],
                    actions=act_stack[:length_n, n],
                    rewards=rew_stack[:length_n, n],
                    behavior_logps=logp_stack[:length_n, n],
                    terminated=bool(fell[n]),  # True = pole fell (a failure)
                )
            )
        return episodes


class CartPoleFamily(TaskFamily):
    """Sequence of CartPole variants.

    Family params (``env.params``):
        gamma: discount factor (default 0.99)
        max_steps: episode truncation horizon (default 200)

    Per-task params (``env.tasks[i]``): any subset of
        length, masspole, masscart, gravity, force_mag
    """

    is_tabular = False

    def __init__(self, params: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        gamma = float(params.get("gamma", 0.99))
        max_steps = int(params.get("max_steps", 200))
        if not tasks:
            raise ValueError("CartPoleFamily needs a non-empty env.tasks list.")

        self.obs_dim = 4
        self.num_actions = 2
        self.tasks = []
        for task_id, task_params in enumerate(tasks):
            label = ",".join(f"{key}={value}" for key, value in task_params.items())
            spec = TaskSpec(task_id, f"cartpole({label or 'default'})", task_params)
            self.tasks.append(CartPoleTask(spec, gamma, max_steps))
