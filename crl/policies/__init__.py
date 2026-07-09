"""Policy registry and construction from config."""

from __future__ import annotations

from crl.config import PolicyConfig
from crl.envs.base import TaskFamily
from crl.policies.base import Policy, clone_policy
from crl.policies.mlp import MLPPolicy, MultiHeadMLPPolicy
from crl.policies.tabular import TabularPolicy


def make_policy(cfg: PolicyConfig, family: TaskFamily) -> Policy:
    """Instantiate the policy named in the config for the given family."""
    if cfg.kind == "tabular":
        return TabularPolicy(family.obs_dim, family.num_actions)
    if cfg.kind == "mlp":
        return MLPPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            task_conditioned=cfg.task_conditioned,
            num_tasks=len(family),
        )
    if cfg.kind == "multihead":
        return MultiHeadMLPPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    raise KeyError(
        f"Unknown policy kind '{cfg.kind}'; available: tabular, mlp, multihead"
    )


__all__ = [
    "Policy", "clone_policy", "make_policy",
    "MLPPolicy", "MultiHeadMLPPolicy", "TabularPolicy",
]
