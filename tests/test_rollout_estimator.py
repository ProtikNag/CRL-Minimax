"""Monte-Carlo estimator agreement with the exact backend on the gridworld.

These are statistical tests with fixed seeds: values must agree within
sampling tolerance, and the REINFORCE gradient must point in the same
direction as the exact policy gradient.
"""

import torch

from crl.config import EnvConfig
from crl.envs import make_family
from crl.estimators.exact import ExactEstimator
from crl.estimators.rollout import MonteCarloEstimator
from crl.policies.tabular import TabularPolicy
from crl.seeding import set_seed


def _setup():
    set_seed(1)
    cfg = EnvConfig(
        family="gridworld",
        params={"size": 4, "slip": 0.1, "gamma": 0.9, "max_steps": 60},
        tasks=[{"goal": [3, 3]}],
    )
    family = make_family(cfg)
    policy = TabularPolicy(family.obs_dim, family.num_actions)
    with torch.no_grad():
        policy.logits.add_(0.3 * torch.randn_like(policy.logits))
    return family.tasks[0], policy


def test_monte_carlo_value_matches_exact():
    task, policy = _setup()
    exact_value = ExactEstimator().evaluate(policy, task)
    mc_value = MonteCarloEstimator(episodes_per_eval=600).evaluate(policy, task)
    # Value scale here is ~0.2-0.5; 0.05 absolute is ~3 sigma at 600 episodes.
    assert abs(mc_value - exact_value) < 0.05


def test_monte_carlo_gradient_direction_matches_exact():
    task, policy = _setup()

    objective, _, _ = ExactEstimator().surrogate_objective(policy, task)
    objective.backward()
    exact_grad = policy.logits.grad.flatten().clone()
    policy.logits.grad = None

    estimator = MonteCarloEstimator(
        episodes_per_grad=600, baseline="batch_mean", time_discount_weighting=True
    )
    mc_objective, _, _ = estimator.surrogate_objective(policy, task)
    mc_objective.backward()
    mc_grad = policy.logits.grad.flatten().clone()

    cosine = torch.nn.functional.cosine_similarity(exact_grad, mc_grad, dim=0)
    assert float(cosine) > 0.5, f"gradient cosine too low: {float(cosine):.3f}"
