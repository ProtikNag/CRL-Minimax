# Handoff

State of the project and what to do next, for a fresh session. Read `README.md`
first (problem, method, math); this file is status + next steps only.

## ►► START HERE

The **gridworld tier is done and validated** (10 tasks, 5 seeds, real rollouts —
`reports/gridworld_tentask/`). Work in progress and next up:

1. **CartPole (in progress).** Six physics-shifted tasks, multi-head policy,
   sampled estimator (`configs/cartpole_multihead.yaml`). Comparison is
   constrained vs fine-tune vs joint (ablation dropped for now). Performance is
   reported as balancing steps / success rate, not value. Batched torch rollout
   (`CartPoleTask.vector_rollout`, verified to match gym exactly) keeps it fast.
   ```bash
   sbatch scripts/hpc_cartpole.sbatch   # array over seeds, 3 methods
   python -m experiments.aggregate_seeds --config configs/cartpole_multihead.yaml \
       --name cartpole_multihead --seeds 0 1 2 --methods constrained finetune joint
   ```
2. **MiniGrid** — a new env family under `crl/envs/` (add a batched rollout like
   the two existing envs), then a 3-game **Atari** subset (needs an actor-critic
   estimator in `crl/estimators/`).
3. **Constraint-strength ablation** (paper Fig 4): sweep `eps` / `duals.lr` with
   `experiments/sweep.py`. Deferred until the method is proven worth it.

## Where things stand

Method from `docs/Objective_for_Continual_Reinforcement_Learning.pdf` fully
implemented. 25/25 tests pass (`pytest -q`, ~1.5 min CPU).

**Gridworld headline** (`configs/gridworld_tentask_sampled.yaml`, multi-head MLP,
sampled estimator, 5 seeds) — mean final value over 10 tasks (±std), success rate:
- constrained (ours): **0.848 ± 0.027**, 99.6% — retains all 10
- joint upper bound: 0.876 ± 0.003, 100% (4 seeds; seed 0 hit the wall limit)
- unconstrained ablation: 0.759 ± 0.039, 95% — forgets NEWEST (T10 → 0.27)
- naive fine-tuning: 0.542 ± 0.093, 79% — forgets OLDEST tasks

The two baselines fail in opposite directions; ours matches the upper bound. This
is the core claim, with error bars.

**Pipeline:** `experiments/multiseed_comparison.py` (one seed → N methods,
`--methods` selectable, SLURM-array friendly) + `experiments/aggregate_seeds.py`
+ `analysis/aggregate.py` (mean ± 95% CI figures/tables). Raw runs → `results/`
(gitignored); committed bundles → `reports/` (tracked).

**Speed:** both envs expose a batched `vector_rollout` (lockstep episode stepping,
one policy forward per timestep) that the MC estimator uses automatically — the
reason the sampled tiers are affordable. Both verified unbiased/exact vs their
reference (exact DP for gridworld; gym step-for-step for CartPole).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
CPU torch is fine for gridworld/CartPole; no GPU needed until the Atari tier. On
the cluster, CPU partition `defq-64core` is used (see `scripts/hpc_*.sbatch`).

## Method facts worth re-checking against an older mental model

- Constraint is a per-task, one-sided, **squared** hinge (eqs 7, 11): penalty
  only where the trained policy is *below* its frozen reference. Local carries
  one `λ_i` per past task; global carries a single `μ`.
- Uniform weights `ω_i = 1/k`. **Replay-free via env access** (fresh rollouts in
  old envs, no buffer). `ε` is in **squared-value units** (`ε = g²`), so its
  scale is env-specific: gridworld `duals.lr: 8`, `eps ~0.002` (value ~1);
  CartPole `duals.lr: 0.003`, `eps ~16` (value ~0-95).

## Open issues

1. **Alternation stability.** Each phase moves the other's frozen reference; the
   pair can cycle. `gaps` figure is always logged. No convergence theory yet —
   the intended research contribution. If cycling appears: shorter phases,
   Polyak-averaged references, or the PID dual (`duals.kind: pid`).
2. **Feasibility / dual saturation.** Tight `ε` can be infeasible when tasks
   conflict in shared states; `λ`/`μ` then ride their `max_value` cap. Multi-head
   + task-conditioned policy (used everywhere) is the main mitigation; loosen
   `eps` or lower `duals.lr` if a dual pins to its cap.
3. **Off-policy stub.** If env re-instantiation ever becomes unavailable, the
   route is a CLEAR-style buffer + fitted-Q or V-trace; a `frozen_surrogate`
   stub exists (`crl/estimators/surrogate.py`). **Do not build it without
   asking** — paper-level design decision, unnecessary under the env-access
   assumption.

## Adding things (registry-driven)

- **Env family:** subclass `TaskFamily` in `crl/envs/`, register in
  `FAMILY_REGISTRY`. Add a batched `vector_rollout` on the task for speed.
  Tabular families also expose `(P, r, ρ)` to unlock the exact estimator. Set
  `success_on_termination` correctly (True = reach a goal; False = survive).
- **Estimator** (e.g. actor-critic): implement `evaluate` + `surrogate_objective`
  in `crl/estimators/`, register in `ESTIMATOR_REGISTRY`. Trainer unchanged.
- **Policy / dual controller:** same pattern under `crl/policies/`, `crl/duals/`.

Keep hyperparameters in `configs/` (no magic numbers in scripts); the config
snapshot is logged with every run automatically.

## Generating figures

- Multi-seed CI bundle (the current headline path):
  `python -m experiments.aggregate_seeds --config <cfg> --name <name> --seeds ...`
- Single-run diagnostics: `python -m analysis.plots --run results/<run_dir>`

Figures are written in PNG + SVG (green = ours, red = fine-tune baseline).
`reports/` is tracked so figures committed on the cluster can be pulled locally.
