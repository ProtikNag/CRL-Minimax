"""Monte-Carlo evaluation for the PPO backend.

The continual-learning constraint uses a **return**-based value estimate (as in
the document / the REINFORCE backend), not the critic. This module rolls out full
episodes under a policy and reports, from the *same* rollout, two quantities:

* ``value``  -- mean discounted return on the *training* reward scale (sign-clipped
                iff the task clips rewards). This is V_i^pi used for the shortfall
                F-hat, the frozen references, and the eval matrix's constraint math.
* ``score``  -- mean *raw* (unclipped) episode return: the reported game score.

Both come from one rollout on an unclipped env, re-deriving the clipped value with
``sign()`` -- this avoids any dependence on vectorized-env info plumbing.

Actions are sampled (not greedy), matching the existing evaluation convention.
"""

from __future__ import annotations

import torch

from crl.envs.base import Task
from crl.policies.base import Policy


@torch.no_grad()
def evaluate_value_and_score(
    policy: Policy,
    task: Task,
    num_episodes: int,
    n_envs: int,
    device: torch.device,
    seed: int | None = None,
    max_env_steps: int = 100_000,
) -> tuple[float, float, int]:
    """Return ``(mean_value, mean_score, n_episodes_used)`` for ``policy`` on ``task``.

    ``mean_value`` is the discounted return on the training reward scale;
    ``mean_score`` is the mean raw undiscounted episode return (game score).
    """
    clip = bool(getattr(task, "clip_rewards", False))
    gamma = task.gamma
    tid = task.spec.task_id
    venv = task.make_vector_env(n_envs, clip_rewards=False)  # raw rewards
    try:
        obs, _ = venv.reset(seed=seed)
        obs = torch.as_tensor(obs, device=device)
        disc = torch.zeros(n_envs)  # running discounted (clipped-scale) return
        gpow = torch.ones(n_envs)  # gamma^t per env
        raw = torch.zeros(n_envs)  # running raw return
        values: list[float] = []
        scores: list[float] = []

        steps = 0
        while len(values) < num_episodes and steps < max_env_steps:
            steps += 1
            dist = policy.dist(obs, tid)
            action = dist.sample().to("cpu").numpy()
            obs_np, reward, term, trunc, _ = venv.step(action)
            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            train_r = torch.sign(reward_t) if clip else reward_t
            disc += gpow * train_r
            raw += reward_t
            gpow *= gamma
            done = term | trunc
            for i in range(n_envs):
                if bool(done[i]):
                    values.append(float(disc[i]))
                    scores.append(float(raw[i]))
                    disc[i] = 0.0
                    raw[i] = 0.0
                    gpow[i] = 1.0
            obs = torch.as_tensor(obs_np, device=device)
    finally:
        venv.close()

    if not values:  # no episode finished within the step budget
        return 0.0, 0.0, 0
    n = min(len(values), num_episodes)
    mean_value = sum(values[:n]) / n
    mean_score = sum(scores[:n]) / n
    return mean_value, mean_score, n
