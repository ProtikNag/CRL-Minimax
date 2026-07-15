"""Vectorized rollout collection and GAE(lambda).

A :class:`RolloutCollector` owns one task's vectorized environment bank and a
persistent ``(next_obs, next_done)`` stream, so successive ``collect`` calls form
one continuous PPO rollout (the standard streaming pattern). Each call gathers
``n_steps`` transitions per env under the *current* policy (which serves as
``pi_old``: its log-probs and value predictions are frozen into the batch), then
computes GAE(lambda) advantages and bootstrapped returns.

The env uses ``SAME_STEP`` autoreset, so masking the bootstrap on ``done`` is the
exact classic formulation -- there are no dummy autoreset transitions to filter.
Truncation is treated like termination for bootstrapping (no value bootstrap on
either); with long Atari horizons this bias is negligible and matches CleanRL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crl.envs.base import Task
from crl.policies.base import Policy


@dataclass
class RolloutBatch:
    """One flattened on-policy batch (``N = n_steps * n_envs`` transitions)."""

    obs: torch.Tensor  # [N, C, H, W] uint8 (policy normalizes to [0,1])
    actions: torch.Tensor  # [N] long
    logprobs: torch.Tensor  # [N] float  log pi_old(a|s)
    advantages: torch.Tensor  # [N] float  GAE
    returns: torch.Tensor  # [N] float   advantages + values (critic target)
    values: torch.Tensor  # [N] float    V_old(s)
    task_id: int
    # Diagnostics from episodes that finished during collection (clipped-reward
    # scale -- a cheap training-progress proxy, not the reported game score).
    ep_returns: list[float]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) over a ``[T, n_envs]`` rollout.

    ``dones[t]`` marks whether ``obs[t]`` began a fresh episode (post-reset), i.e.
    the CleanRL convention where ``dones[t+1]`` gates the bootstrap out of step t.
    Returns ``(advantages, returns)`` each ``[T, n_envs]``.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutCollector:
    """Persistent vectorized-env stream for one task."""

    def __init__(
        self,
        task: Task,
        n_envs: int,
        n_steps: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.task_id = task.spec.task_id
        self.gamma = task.gamma
        self.n_envs = int(n_envs)
        self.n_steps = int(n_steps)
        self.device = device
        self.venv = task.make_vector_env(self.n_envs)
        obs, _ = self.venv.reset(seed=seed)
        self.next_obs = torch.as_tensor(obs, device=device)  # uint8 [n_envs,C,H,W]
        self.next_done = torch.zeros(self.n_envs, device=device)
        # Running clipped-return accumulator per env (for progress diagnostics).
        self._ep_ret = torch.zeros(self.n_envs)

    def close(self) -> None:
        self.venv.close()

    @torch.no_grad()
    def collect(self, policy: Policy, gae_lambda: float) -> RolloutBatch:
        """Roll out ``n_steps`` under ``policy`` (as pi_old); return a batch."""
        device, tid = self.device, self.task_id
        T, n = self.n_steps, self.n_envs
        C, H, W = self.next_obs.shape[1:]

        obs_buf = torch.empty((T, n, C, H, W), dtype=torch.uint8, device=device)
        act_buf = torch.empty((T, n), dtype=torch.long, device=device)
        logp_buf = torch.empty((T, n), device=device)
        rew_buf = torch.empty((T, n), device=device)
        done_buf = torch.empty((T, n), device=device)
        val_buf = torch.empty((T, n), device=device)
        ep_returns: list[float] = []

        for t in range(T):
            obs_buf[t] = self.next_obs
            done_buf[t] = self.next_done
            dist, value = policy.dist_value(self.next_obs, tid)
            action = dist.sample()
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value

            action_cpu = action.to("cpu").numpy()
            obs, reward, term, trunc, _ = self.venv.step(action_cpu)
            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            rew_buf[t] = reward_t.to(device)
            done_np = term | trunc
            self.next_obs = torch.as_tensor(obs, device=device)
            self.next_done = torch.as_tensor(
                done_np, dtype=torch.float32, device=device
            )
            # Track finished episodes (clipped-reward return) for logging.
            self._ep_ret += reward_t
            for i in range(n):
                if bool(done_np[i]):
                    ep_returns.append(float(self._ep_ret[i]))
                    self._ep_ret[i] = 0.0

        next_value = policy.value(self.next_obs, tid)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, self.next_done,
            self.gamma, gae_lambda,
        )
        return RolloutBatch(
            obs=obs_buf.reshape(T * n, C, H, W),
            actions=act_buf.reshape(T * n),
            logprobs=logp_buf.reshape(T * n),
            advantages=advantages.reshape(T * n),
            returns=returns.reshape(T * n),
            values=val_buf.reshape(T * n),
            task_id=tid,
            ep_returns=ep_returns,
        )
