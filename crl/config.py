"""Typed configuration loaded from YAML.

Every hyperparameter of a run lives in one YAML file under ``configs/``.
Unknown keys raise immediately so silent typos cannot corrupt a sweep.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Run identity and bookkeeping."""

    name: str = "run"
    seed: int = 42
    results_dir: str = "results"
    device: str = "cpu"
    log_every: int = 10


@dataclass
class EnvConfig:
    """Task-family selection.

    ``family`` picks an entry from :data:`crl.envs.FAMILY_REGISTRY`;
    ``params`` are family-wide settings; ``tasks`` is an ordered list of
    per-task parameter dicts (one dict per task, arrival order = list order).
    """

    family: str = "gridworld"
    params: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Policy architecture."""

    kind: str = "tabular"  # tabular | mlp
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    # Append a one-hot task id to observations (decide early; README risk 2).
    task_conditioned: bool = False


@dataclass
class EstimatorConfig:
    """Value / gradient estimation backend.

    ``kind`` picks from :data:`crl.estimators.ESTIMATOR_REGISTRY`:
      exact        -- tabular dynamic-programming evaluation, zero variance
      monte_carlo  -- on-policy episode rollouts (REINFORCE-style)
    """

    kind: str = "exact"
    episodes_per_eval: int = 16  # episodes for a scalar value estimate
    episodes_per_grad: int = 16  # episodes for one gradient estimate
    episodes_per_ref: int = 64  # frozen-reference values (once per phase)
    baseline: str = "batch_mean"  # batch_mean | none
    # Keep the gamma^t weight on the REINFORCE terms so the estimator matches
    # the discounted policy gradient exactly (tested against `exact`).
    time_discount_weighting: bool = True


@dataclass
class DualConfig:
    """Dual-variable controller (lambda and mu use identical settings)."""

    kind: str = "projected_ascent"  # projected_ascent | pid
    lr: float = 0.05
    init: float = 0.0
    max_value: float = 100.0
    # Keep the dual value across phases/tasks instead of resetting to init.
    warm_start: bool = True
    # PID gains (used only when kind == "pid"; Stooke et al. 2020).
    kp: float = 0.05
    ki: float = 0.01
    kd: float = 0.0


@dataclass
class TrainerConfig:
    """Alternation schedule and primal-dual updates (eqs 22-24 and 30-32).

    The past-task constraint is a per-task, one-sided, *squared* hinge
    (eqs 7, 11): a penalty applies only where the trained policy falls below
    its frozen reference on a task (forgetting), and is zero when it is at or
    above the reference. The local phase therefore carries one multiplier
    lambda_i per past task; the global phase carries a single mu on the
    current task.
    """

    cycles_per_task: int = 2  # local<->global alternations per task
    local_steps: int = 50  # primal steps per local phase
    global_steps: int = 50  # primal steps per global phase
    task1_steps: int = 200  # plain ascent on task 1 (no past tasks yet)
    lr_local: float = 0.05  # alpha in eq 22
    lr_global: float = 0.05  # beta in eq 30
    optimizer: str = "sgd"  # sgd | adam  (sgd matches the derivation)
    # Constraint tolerance epsilon. IMPORTANT: units are squared value
    # (the constraint is a squared shortfall), so a tolerated value gap of g
    # corresponds to eps = g**2. A scalar is broadcast to every past task
    # (eps_i = eps in eq 8); a list sets eps_i per past task (length k-1 at
    # the final task, indexed by task arrival order).
    eps: float = 0.0025
    omega: list[float] | None = None  # per-task weights; None = uniform 1/k
    # all: sum over every past task per step (matches the derivation exactly).
    # sample: one past task per step, contribution rescaled by the past-task
    # count -- an unbiased O(1)-cost variant for large task sequences.
    past_task_sampling: str = "all"
    entropy_coef: float = 0.0
    eval_episodes: int = 32  # end-of-task evaluation matrix
    eval_all_tasks: bool = True  # include future tasks in the matrix
    # Probe the global policy on every task this often (in cumulative primal
    # steps) to build learning-curve data; 0 disables probing.
    eval_probe_every: int = 25
    # Report the *undiscounted* return (task performance / game score) in the
    # eval matrix and probes, instead of the discounted value. The constraint
    # and objective still use discounted value; only reporting changes. Set for
    # environments where the paper metric is the score (e.g. MinAtar).
    report_return: bool = False


@dataclass
class Config:
    """Top-level run configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    duals: DualConfig = field(default_factory=DualConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (for logging alongside results)."""
        return dataclasses.asdict(self)


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys at '{path}': {sorted(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a :class:`Config` from a nested dict with strict key checking."""
    sections = {
        "experiment": ExperimentConfig,
        "env": EnvConfig,
        "policy": PolicyConfig,
        "estimator": EstimatorConfig,
        "duals": DualConfig,
        "trainer": TrainerConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise KeyError(f"Unknown top-level config sections: {sorted(unknown)}")
    kwargs = {
        name: _build(cls, raw.get(name, {}) or {}, name)
        for name, cls in sections.items()
    }
    return Config(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)
