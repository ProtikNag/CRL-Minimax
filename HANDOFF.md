# Handoff

State of the project and what to do next, for a fresh session. Read `README.md`
first (problem, method, math); this file is status + next steps only.

## ►► AUTONOMOUS ATARI-PPO RUN IN FLIGHT (2026-07-15)

The active study is the **PPO port on 5 Atari games** (`trainer.kind: ppo`) —
same min-max formulation, PPO instead of REINFORCE (`docs/REINFORCE_to_PPO.md`).
Games: **Pong → Breakout → Boxing → Qbert → SpaceInvaders** (Freeway was dropped;
sparse-reward, never learned). Multi-head actor-critic (shared Nature-CNN trunk +
per-task actor+critic heads), 18-action set.

**Headline = `atari5_ppo_v4`**, 5 seeds × {constrained, finetune}, jobs
`21679631..21679640` + aggregator `21679641` (afterany, `--seeds 0 1 2 3 4`,
`scripts/hpc_atari_aggregate.sbatch` → commits `reports/atari5_ppo_v4/`). Launched
with `scripts/hpc_atari_worker.sbatch configs/atari5_ppo_v4{,_finetune}.yaml <seed>`.
GPU partitions `gpu,gpu-v100-16gb,gpu-v100-32gb` (V100 nodes can preempt/requeue;
`gpu`/node242 is stable).

**When you return, check:** `bash scripts/atari_status.sh` (v1-name only) or read
`results/atari5_ppo_v4_*_seed*/logs.jsonl` (flushed; stdout is block-buffered).
A run is done when `eval_matrix.json` has 5 rows. If a job was preempted (logs
restart from task 1), resubmit it. Then verify `reports/atari5_ppo_v4/tables/`
(`cl_metrics.csv` = AvgPerf/Forgetting/BWT raw+normalized; `retention_table.csv`;
`score_table.csv`) got committed; if the aggregator failed, run
`python -m experiments.aggregate_atari --name atari5_ppo_v4 --seeds 0 1 2 3 4`.

**v4 design (reviewer-facing):** equal per-**model** budget — local, global,
finetune each get the same iteration cap and per-game **greedy threshold**, and
each early-stops when it clears the threshold (`ppo.patience` checks). Reported
scores use **greedy actions, 50 episodes, fixed seed** (low variance, raw kept);
the constraint's V stays on-policy stochastic. Thresholds live in the config
(`env.tasks[i].threshold`): Pong 18, Breakout 50, Boxing 90, Qbert 2000,
SpaceInvaders 600.

**Critical fix baked into v4** (`crl/ppo/trainer.py::optimize_batches`): normalize
the actor coefficients by their sum. Without it, a saturated dual μ makes the
current-task actor gradient dominate the shared grad-norm clip and **starve the
shared critic** → broken GAE → the global cannot consolidate (V_k stuck far below
the local reference). Normalizing is a positive rescaling of the primal direction,
so the KKT fixed point is unchanged. See `docs/REINFORCE_to_PPO.md` §3.

The REINFORCE/MinAtar path is unchanged and remains the theory double-check
(`configs/minatar_*.yaml`, `experiments/aggregate_theory.py`).

## Where things stand

Method from `docs/Objective_for_Continual_Reinforcement_Learning.pdf` fully
implemented and **double-checked against the derivation** — including the fix to
use `ω_i = 1/k` (current task count) per the Setup, not a fixed `1/num_tasks`.
The **PPO/Atari backend** (`crl/ppo/`, `crl/ppo_continual.py`) implements the same
formulation with PPO as the optimizer. 30/30 tests pass (`pytest -q`, ~1.7 min CPU).

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
