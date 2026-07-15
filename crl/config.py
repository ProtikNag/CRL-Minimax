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
    # Optimizer backend / trainer variant. "alternation" = the REINFORCE
    # primal-dual trainer (crl.trainer.AlternationTrainer). "ppo" = the PPO
    # backend (crl.ppo_continual.PPOAlternationTrainer): PPO replaces REINFORCE
    # as the optimizer, the CL framework is unchanged, and PPO settings come from
    # the ``ppo`` config section.
    kind: str = "alternation"
    # Variant: drop the past-task constraint from the LOCAL phase, so the local
    # policy is a pure-plasticity learner of the current task (maximizes V_k
    # only, no lambda shortfall terms). The GLOBAL phase is unchanged -- it is
    # still constrained w.r.t. the frozen local via mu. This tests whether
    # letting the local fully master the new task (then consolidating into the
    # global) fixes current-task underlearning without losing past retention.
    local_unconstrained: bool = False


@dataclass
class PPOConfig:
    """PPO optimizer settings for the ``trainer.kind == "ppo"`` backend.

    PPO replaces REINFORCE as the policy-gradient *optimizer* only; the
    continual-learning framework (local/global alternation, squared-shortfall
    constraint, lambda/mu duals, replay-free fresh rollouts) is unchanged and is
    configured through :class:`TrainerConfig` / :class:`DualConfig`. The critic,
    GAE, value loss and entropy bonus here are standard PPO -- only the ACTOR
    receives the constraint (in the global phase).
    """

    # Continual method: "constrained" = full local/global min-max consolidation;
    # "finetune" = naive sequential standard PPO on one shared net (the
    # catastrophic-forgetting baseline; no local phase, no constraint).
    method: str = "constrained"
    n_envs: int = 8  # parallel vectorized envs feeding the collector
    n_steps: int = 128  # rollout length per env per PPO iteration
    ppo_epochs: int = 4  # optimization epochs over each collected batch
    num_minibatches: int = 4  # minibatches per epoch (batch = n_envs*n_steps)
    clip_ratio: float = 0.1  # PPO clip epsilon (0.1 is standard for Atari)
    gae_lambda: float = 0.95  # GAE(lambda)
    vf_coef: float = 0.5  # value-loss weight (standard PPO critic)
    ent_coef: float = 0.01  # entropy bonus (standard PPO)
    max_grad_norm: float = 0.5  # global grad-norm clip
    lr: float = 2.5e-4  # Adam learning rate (actor+critic share the trunk)
    normalize_advantage: bool = True  # per-minibatch advantage normalization
    # Phase budgets are now MAX CAPS (each model early-stops at its per-game
    # threshold; see below). Counted in PPO iterations (each = collect+ppo_epochs).
    # For a fair per-MODEL comparison, local, global and finetune all use the
    # SAME cap and the SAME per-game thresholds.
    task1_iters: int = 2000  # cap for plain PPO on task 1
    local_iters: int = 2000  # cap for the local phase (standard PPO on current)
    global_iters: int = 2000  # cap for the constrained global consolidation
    # Early stopping: stop a phase once the current game's GREEDY score is
    # >= its threshold for `patience` consecutive checks (but at least min_iters),
    # else run to the cap. Thresholds are per task (env.tasks[i].threshold).
    min_iters: int = 200  # floor per phase before early-stop can trigger
    patience: int = 3  # consecutive threshold-meeting checks required to stop
    stop_eval_every: int = 50  # iters between early-stop score checks
    stop_eval_episodes: int = 15  # greedy episodes per early-stop check
    # Full-episode Monte-Carlo estimates of V_k for the shortfall / references
    # (paper-faithful: the constraint value is the on-policy STOCHASTIC return).
    constraint_episodes: int = 16
    # Re-estimate V_k^G and update mu every N global iterations (slower dual
    # timescale; between updates the mu*2*shortfall coefficient is held fixed).
    constraint_every: int = 5
    # Reported evaluation (eval matrix, probes): greedy (argmax) actions, many
    # episodes, fixed seed -> low variance, reproducible, respectable scores.
    eval_episodes: int = 50  # episodes for the eval matrix / probes
    eval_greedy: bool = True  # argmax actions for the reported score
    eval_seed: int = 100_000  # fixed base seed for evaluation rollouts
    eval_every: int = 0  # probe the global policy every N cumulative iters (0=off)


@dataclass
class Config:
    """Top-level run configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    duals: DualConfig = field(default_factory=DualConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

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
        "ppo": PPOConfig,
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
