# Handoff

State of the project and what to do next, for a fresh session. Read `README.md`
first (problem, method, math); this file is status + next steps only.

## ►► START HERE

The project is now **MinAtar-only, pure REINFORCE, following the theory exactly**.
Gridworld/tabular/exact are retained solely as the exact-gradient **test harness**
(the theory double-check); CartPole and the value-based (DQN) exploration were
removed to stay within the policy-gradient derivation.

**Current headline run (in flight / latest):** four MinAtar games
(SpaceInvaders → Breakout → Asterix → Seaquest), shared conv trunk + per-task
heads, **5 seeds**, three methods:

- `constrained` — full theory, local constrained on past tasks (ours).
- `localfree` — unconstrained-local variant (ours).
- `finetune` — naive sequential (baseline).

Launch (per seed, 3 jobs):
```bash
sbatch scripts/hpc_minatar.sbatch constrained <seed> configs/minatar_multihead.yaml minatar_multihead
sbatch scripts/hpc_minatar.sbatch finetune    <seed> configs/minatar_multihead.yaml minatar_multihead
sbatch scripts/hpc_minatar.sbatch constrained <seed> configs/minatar_localfree.yaml minatar_localfree
```
Aggregate all seeds into the report bundle:
```bash
python -m experiments.aggregate_theory --seeds 0 1 2 3 4
# -> reports/minatar_theory/{figures,tables}
```

**Next up:** once the 5-seed bundle lands, fill the headline table in
`README.md` (`<!-- HEADLINE_RESULTS -->`) and this file, then decide whether to
scale games / seeds or tune `eps`/`duals.lr`.

## Where things stand

Method from `docs/Objective_for_Continual_Reinforcement_Learning.pdf` fully
implemented and **double-checked against the derivation** — including the fix to
use `ω_i = 1/k` (current task count) per the Setup, not a fixed `1/num_tasks`.
25/25 tests pass (`pytest -q`, ~1.5 min CPU).

**Pipeline:** `experiments/multiseed_comparison.py` (one seed → N methods,
`--methods` selectable, SLURM-array friendly) + `experiments/aggregate_theory.py`
(3-method MinAtar CI bundle) + `analysis/aggregate.py` (mean ± 95% CI
figures/tables). Raw runs → `results/` (gitignored); committed bundles →
`reports/` (tracked).

**Speed:** MinAtar exposes a batched `vector_rollout` (lockstep episode stepping,
one policy forward per timestep) that the MC estimator uses automatically — the
reason the sampled tier is affordable on CPU. The gridworld harness rollout is
verified unbiased vs the exact DP estimator.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes minatar
pytest -q
```
CPU torch is fine (MinAtar is CPU-env-bound; the Python env-stepping loop
dominates, not CNN compute). On the cluster, CPU partition `defq-64core` is used
(see `scripts/hpc_minatar.sbatch`).

## Method facts worth re-checking against an older mental model

- Constraint is a per-task, one-sided, **squared** hinge (eqs 7, 11): penalty
  only where the trained policy is *below* its frozen reference. Local carries
  one `λ_i` per past task; global carries a single `μ`.
- Uniform weights **`ω_i = 1/k`** (current task count, shrinks with `k`).
  **Replay-free via env access** (fresh rollouts in old envs, no buffer). `ε` is
  in **squared-value units** (`ε = g²`); MinAtar uses `duals.lr: 1.0`,
  `eps: 0.04` with per-task `reward_scale` making values ~O(1).
- `local_unconstrained: true` gives the unconstrained-local variant (drops `λ`).
- Reporting metric is the **raw game score** (undiscounted return), not value.
- The entropy bonus (`entropy_coef`) and the batch-mean REINFORCE baseline are
  practical add-ons; both leave the policy gradient unbiased (verified vs exact).

## Open issues

1. **Alternation stability.** Each phase moves the other's frozen reference; the
   pair can cycle. `gaps` figure is always logged. No convergence theory yet —
   the intended research contribution. If cycling appears: shorter phases,
   Polyak-averaged references, or the PID dual (`duals.kind: pid`).
2. **Feasibility / dual saturation.** Tight `ε` can be infeasible when tasks
   conflict in shared states; `λ`/`μ` then ride their `max_value` cap. Multi-head
   policy is the main mitigation; loosen `eps` or lower `duals.lr` if a dual pins.
3. **REINFORCE learnability.** Some MinAtar games (Asterix, Seaquest) are hard to
   learn well under pure REINFORCE in a tractable budget. Raise `local_steps` /
   `task1_steps` / `episodes_per_grad` if a game underlearns; report honestly.
4. **Off-policy stub.** If env re-instantiation ever becomes unavailable, the
   route is a CLEAR-style buffer + fitted-Q or V-trace; a `frozen_surrogate` stub
   exists (`crl/estimators/surrogate.py`). **Do not build it without asking.**

## Adding things (registry-driven)

- **Env family:** subclass `TaskFamily` in `crl/envs/`, register in
  `FAMILY_REGISTRY`. Add a batched `vector_rollout` on the task for speed. Set
  `success_on_termination` correctly (True = reach a goal; False = survive/score).
- **Estimator:** implement `evaluate` + `surrogate_objective` in
  `crl/estimators/`, register in `ESTIMATOR_REGISTRY`. Trainer unchanged.
- **Policy / dual controller:** same pattern under `crl/policies/`, `crl/duals/`.

Keep hyperparameters in `configs/` (no magic numbers in scripts); the config
snapshot is logged with every run automatically.

## Generating figures

- MinAtar 3-method CI bundle (current headline path):
  `python -m experiments.aggregate_theory --seeds 0 1 2 3 4`
- Single-run diagnostics: `python -m analysis.plots --run results/<run_dir>`

Figures are written in PNG + SVG. Method colors: green = constrained-local (ours),
blue = unconstrained-local (ours), red = fine-tune baseline. `reports/` is tracked
so figures committed on the cluster can be pulled locally.
```
