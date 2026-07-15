"""Monte-Carlo evaluation for the PPO backend.

Two distinct quantities, from full-episode rollouts:

* ``value``  -- mean discounted return on the *training* reward scale (sign-clipped
                iff the task clips rewards). This is V_i^pi used for the shortfall
                F-hat, the frozen references, and the constraint. It is measured on
                the *stochastic* policy (sampled actions), faithful to the theory's
                V^pi. Both come from one rollout on an unclipped env; the clipped
                value is re-derived with ``sign()`` to avoid info-plumbing.
* ``score``  -- mean *raw* (unclipped) episode return: the reported game score.

For REPORTING (eval matrix, probes, early-stopping thresholds) we use
``greedy=True`` (argmax actions), ``num_episodes`` large (e.g. 50), and a FIXED
``seed`` so the measurement is low-variance and reproducible across methods,
seeds and checkpoints. For the CONSTRAINT value we use ``greedy=False`` (the
on-policy stochastic value). ``std`` (of the raw score across episodes) is
returned for error bars.
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
    greedy: bool = False,
    max_env_steps: int = 200_000,
) -> tuple[float, float, float, int]:
    """Return ``(mean_value, mean_score, score_std, n_episodes_used)``.

    ``mean_value`` = discounted return on the training reward scale (stochastic
    policy semantics regardless of ``greedy`` for the clipped-value accounting).
    ``mean_score`` / ``score_std`` = mean and std of the raw undiscounted episode
    return (game score). ``greedy`` selects argmax actions (for reporting);
    otherwise actions are sampled. A fixed ``seed`` makes the rollout reproducible.
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
        while len(scores) < num_episodes and steps < max_env_steps:
            steps += 1
            logits = policy.dist(obs, tid).logits
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            action = action.to("cpu").numpy()
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

    if not scores:  # no episode finished within the step budget
        return 0.0, 0.0, 0.0, 0
    n = min(len(scores), num_episodes)
    sc = torch.tensor(scores[:n])
    mean_value = sum(values[:n]) / n
    return mean_value, float(sc.mean()), float(sc.std(unbiased=False)), n
