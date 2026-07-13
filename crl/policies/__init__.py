"""Policy registry and construction from config."""

from __future__ import annotations

from crl.config import PolicyConfig
from crl.envs.base import TaskFamily
from crl.policies.base import Policy, clone_policy
from crl.policies.cnn import MinAtarCNNPolicy, MinAtarMultiHeadCNNPolicy
from crl.policies.mlp import MLPPolicy, MultiHeadMLPPolicy
from crl.policies.qnet import MinAtarMultiHeadQNetwork, MinAtarQNetwork
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
    if cfg.kind in ("cnn", "cnn_multihead", "qnet", "qnet_multihead"):
        obs_shape = getattr(family, "obs_shape", None)
        if obs_shape is None:
            raise ValueError(f"policy '{cfg.kind}' needs a family with obs_shape "
                             "(e.g. minatar), not a flat-vector family.")
        hidden = list(cfg.hidden_sizes)[0] if cfg.hidden_sizes else 128
        registry = {
            "cnn": MinAtarCNNPolicy,
            "cnn_multihead": MinAtarMultiHeadCNNPolicy,
            "qnet": MinAtarQNetwork,
            "qnet_multihead": MinAtarMultiHeadQNetwork,
        }
        return registry[cfg.kind](
            obs_shape=obs_shape,
            num_actions=family.num_actions,
            hidden_size=hidden,
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    raise KeyError(
        f"Unknown policy kind '{cfg.kind}'; available: tabular, mlp, multihead, "
        "cnn, cnn_multihead, qnet, qnet_multihead"
    )


__all__ = [
    "Policy", "clone_policy", "make_policy",
    "MLPPolicy", "MultiHeadMLPPolicy", "TabularPolicy",
]
