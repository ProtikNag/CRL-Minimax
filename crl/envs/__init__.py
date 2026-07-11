"""Task-family registry: add a family here to make it config-selectable."""

from __future__ import annotations

from crl.config import EnvConfig
from crl.envs.base import TabularTask, Task, TaskFamily, TaskSpec
from crl.envs.cartpole import CartPoleFamily
from crl.envs.gridworld import GridWorldFamily
from crl.envs.minatar import MinAtarFamily

FAMILY_REGISTRY: dict[str, type[TaskFamily]] = {
    "gridworld": GridWorldFamily,
    "cartpole": CartPoleFamily,
    "minatar": MinAtarFamily,
}


def make_family(cfg: EnvConfig) -> TaskFamily:
    """Instantiate the task family named in the config."""
    if cfg.family not in FAMILY_REGISTRY:
        raise KeyError(
            f"Unknown env family '{cfg.family}'; available: {sorted(FAMILY_REGISTRY)}"
        )
    return FAMILY_REGISTRY[cfg.family](cfg.params, cfg.tasks)


__all__ = [
    "FAMILY_REGISTRY",
    "make_family",
    "Task",
    "TabularTask",
    "TaskFamily",
    "TaskSpec",
]
