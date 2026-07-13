"""Convolutional MinAtar Q-networks: shared conv trunk + Q head(s).

Value-based counterpart of ``crl/policies/cnn.py``. The head outputs one
Q-value per action (not action logits); the induced policy is greedy,
a = argmax_a Q(s, a). Used by the Double-DQN continual learner (``crl/dqn.py``)
where the min-max theory's value-gradient steps are realized as DDQN steps.

Two variants mirror the policy-gradient nets:
    MinAtarQNetwork            single shared Q head (unified action set)
    MinAtarMultiHeadQNetwork   shared conv+FC trunk, one Q head per task

The shared conv trunk is where cross-game interference (forgetting) lives; the
constraint on the global network protects it, exactly as in the PG method.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy
from crl.policies.cnn import _CNNTrunk


class _QNetMixin:
    """Greedy action selection + a degenerate ``dist`` for the Policy ABC.

    The DDQN learner never samples from ``dist``; it calls ``q_values``. But
    ``rollout_performance`` (the reporting path) calls ``policy.act``, which we
    override to be greedy (epsilon = 0). ``dist`` is implemented only to satisfy
    the abstract interface and returns a distribution peaked at the greedy
    action so any accidental sampling is still greedy.
    """

    def q_values(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        raise NotImplementedError

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        # Large temperature -> effectively deterministic (argmax) sampling.
        return Categorical(logits=self.q_values(obs, task_id) * 1e6)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, task_id: int) -> int:
        """Greedy action for a single observation ``[C, H, W]``."""
        q = self.q_values(obs.unsqueeze(0), task_id)
        return int(q.argmax(dim=-1).item())


class MinAtarQNetwork(_QNetMixin, Policy):
    """Shared conv trunk + a single shared Q head (unified action set)."""

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int,
                 hidden_size: int = 128, num_tasks: int = 0,
                 task_conditioned: bool = False) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.trunk = _CNNTrunk(c, h, w, hidden_size, num_tasks, task_conditioned)
        self.head = nn.Linear(hidden_size, num_actions)

    def q_values(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.head(self.trunk(obs, task_id))


class MinAtarMultiHeadQNetwork(_QNetMixin, Policy):
    """Shared conv+FC trunk with one Q head per task (hard task routing)."""

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int,
                 hidden_size: int = 128, num_tasks: int = 0,
                 task_conditioned: bool = False) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MinAtarMultiHeadQNetwork requires num_tasks > 0.")
        c, h, w = obs_shape
        self.num_tasks = num_tasks
        self.trunk = _CNNTrunk(c, h, w, hidden_size, num_tasks, task_conditioned)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_actions) for _ in range(num_tasks)])

    def q_values(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        return self.heads[task_id](self.trunk(obs, task_id))
