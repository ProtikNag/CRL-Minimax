"""MLP categorical policy, optionally task-conditioned."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


class MLPPolicy(Policy):
    """Feed-forward categorical policy.

    When ``task_conditioned`` is true a one-hot task indicator is appended to
    every observation, letting one network represent task-specific behavior
    in shared states (README risk 2: feasibility under task conflict).
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
        layers: list[nn.Module] = []
        last = input_dim
        for width in hidden_sizes:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        head = nn.Linear(last, num_actions)
        # Small final layer keeps the initial policy near-uniform, which keeps
        # early REINFORCE gradients low-variance.
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        if self.task_conditioned:
            one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device)
            one_hot[:, task_id] = 1.0
            obs = torch.cat([obs, one_hot], dim=-1)
        return Categorical(logits=self.net(obs))
