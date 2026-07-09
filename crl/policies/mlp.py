"""MLP categorical policies: shared-body and multi-head variants."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


def _mlp_trunk(input_dim: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
    """Tanh MLP trunk; returns the module and its output width."""
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.Tanh()]
        last = width
    return nn.Sequential(*layers), last


def _init_head(head: nn.Linear) -> None:
    """Small final layer keeps the initial policy near-uniform (low-variance
    early gradients)."""
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)


class MLPPolicy(Policy):
    """Single-head feed-forward categorical policy.

    When ``task_conditioned`` is true a one-hot task indicator is appended to
    every observation, letting one shared network represent task-specific
    behavior in shared states.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        task_conditioned: bool = False,
        num_tasks: int = 0,
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks

        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.head = nn.Linear(last, num_actions)
        _init_head(self.head)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.head(self.trunk(self._augment(obs, task_id))))


class MultiHeadMLPPolicy(Policy):
    """Shared trunk with one output head per task (hard task routing).

    Each task owns a separate linear head on top of a shared Tanh trunk, so a
    task never has to share its output mapping with a conflicting task -- the
    task identifier selects the head. The trunk is shared (that is where
    forward transfer and, without the constraint, forgetting live); the
    min-max constraint is what protects the shared trunk. Optionally the task
    one-hot is also appended to the trunk input (``task_conditioned``), giving
    the trunk explicit task awareness on top of the routed heads.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MultiHeadMLPPolicy requires num_tasks > 0.")
        self.num_tasks = num_tasks
        self.task_conditioned = task_conditioned

        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.heads = nn.ModuleList(
            [nn.Linear(last, num_actions) for _ in range(num_tasks)]
        )
        for head in self.heads:
            _init_head(head)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        features = self.trunk(self._augment(obs, task_id))
        return Categorical(logits=self.heads[task_id](features))
