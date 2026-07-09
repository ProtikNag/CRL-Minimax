"""Parametric CartPole family (development tier, sampled estimators only).

Each task overrides physics parameters of ``CartPole-v1`` (pole length, pole
mass, gravity, force magnitude). Dynamics change across tasks while state and
action spaces stay fixed, giving a cheap non-tabular testbed for the sampled
estimators and the alternation loop.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from crl.envs.base import Task, TaskFamily, TaskSpec

# Physics attributes that may be overridden per task.
_PHYSICS_KEYS = ("length", "masspole", "masscart", "gravity", "force_mag")


class CartPoleTask(Task):
    """CartPole-v1 with per-task physics overrides."""

    def __init__(self, spec: TaskSpec, gamma: float, max_steps: int) -> None:
        super().__init__(spec, gamma)
        self._max_steps = max_steps

    def make_env(self) -> gym.Env:
        env = gym.make("CartPole-v1", max_episode_steps=self._max_steps)
        core = env.unwrapped
        for key, value in self.spec.params.items():
            if key not in _PHYSICS_KEYS:
                raise KeyError(
                    f"Unknown CartPole physics key '{key}'; allowed: {_PHYSICS_KEYS}"
                )
            setattr(core, key, float(value))
        # CartPoleEnv derives these in __init__; recompute after overrides.
        core.total_mass = core.masspole + core.masscart
        core.polemass_length = core.masspole * core.length
        return env


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
