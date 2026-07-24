# Handoff

State of the project and what to do next, for a fresh session. Read `README.md`
first (problem, method, math); this file is status + next steps only.

---

## ►► CURRENT STATE (2026-07-23)

The project has moved from "does the min-max method work on Atari?" (yes, ~ties
CLEAR on forgetting) to **"why does it under-consolidate, and what's the fix?"**
Two things are live:

1. **Precomputed single-task experts are training** (Impala-CNN, 10 games) — see
   *IN FLIGHT* below. These become **fixed local references** reused by every
   future consolidation experiment (train once, stop re-training locals).
2. A batch of **diagnostics experiments is complete** and located the problem —
   see *Findings* below and `diagnostics/README.md`.

### Binding project rules (do not violate)

- **Greedy-only evaluation, 100 rollouts, never stochastic.** All reported scores
  (eval matrix, retention, probes, expert curves) use argmax actions, 100
  episodes. Config default is `eval_episodes: 100`, `eval_greedy: true`,
  `eval_greedy_episodes ≥ eval_episodes`. The retired modes (200greedy+100stoch,
  "random 150" all-stochastic) are gone. The *constraint* value `V_G/V_L`
  (`constraint_episodes`) stays on-policy stochastic — it's the constrained
  quantity, not "evaluation." (memory: `greedy-only-evaluation`)
- **No per-episode step cap: `max_steps: 0`.** ALE-v5 imposes no internal limit,
  so a `TimeLimit` is the only truncation source — and with `SAME_STEP` autoreset
  the collector's `done = term | trunc` bootstraps **0 future value** at the cap,
  poisoning GAE on long episodes. A 4000-step cap silently broke Breakout
  training. Keep envs uncapped everywhere (training, eval, and the future
  consolidation harness). Eval step-budget scales with `num_episodes` so
  greedy-100 still completes on long-episode agents (`crl/ppo/evaluate.py`).

### Findings (what the diagnostics established) — `diagnostics/`

- **v5 full-budget min-max vs CLEAR (n=2, greedy):** AvgPerf 0.45 vs **0.73**,
  Forgetting **0.44** vs 0.46, BWT **−0.40** vs −0.46. Min-max ~ties CLEAR on
  forgetting/BWT (replay-free); CLEAR wins absolute performance.
  Figures: `reports/atari5_ppo_v4/figures/{clear_comparison,constrained}/`.
- **μ pinning was a symptom, not the cause.** `warm_start=false` stopped μ
  ratcheting to its ceiling, but the **current-task gradient still dominates**
  (‖g_new‖/‖g_old‖ = 5–34× when the constraint is active) because
  `coeff_k = μ·2·shortfall` is large whenever the current task is hard.
  (`diagnostics/atari_diag_seed0/`, `gradient_conflict/`)
- **EXP1 feasibility (`diagnostics/feasibility/`):** even a *joint* model (all
  games at once, no constraint, equal budget) can't fit all games — Breakout
  monopolizes the trunk, Pong/Boxing collapse. Shared-trunk **interference is
  real** (capacity, H3). Caveat: joint was undertrained (1200 iters) — the
  higher-budget joint is the clean confirmation (next step 1).
- **EXP2 value-constraint sufficiency (`diagnostics/value_constraint/`):** the
  scalar `V_G ≥ V_L` is **too weak** — value gap → 0 while KL(π_local‖π_global)
  stays ~0.8 (behaviorally different policy satisfies the number). Adding a
  behavioral-cloning term (`global_bc_coef=0.1`) **collapses KL to ~0.04 and
  helps the value gap close on hard tasks.** Behavior-matching is the fix. (H2)
- **Head-only probe (`diagnostics/head_only_probe/`):** freezing the trunk and
  consolidating only the heads *retains* the current task → the damage in the
  normal method is in **shared-representation (trunk) updates**.

**Synthesis:** two compounding problems — (H3) shared-trunk interference and
(H2) the value constraint is an insufficient statistic for behavior. The fix that
the evidence points to is a **behavior-level constraint** (policy-KL now;
occupancy / successor-feature matching as the rigorous, publishable version),
plus (conditionally) more/modular capacity.

---

## ►► IN FLIGHT: precomputed single-task experts (jobs 21705132–141)

Train each game's expert **once** with unconstrained PPO to convergence, then
reuse the frozen model as the **fixed local reference** (its value `V` for the
constraint, its policy `π` for behavioral cloning) in every consolidation
experiment. This deviates from `θ⁰=φ` (experts start from a shared init, not the
evolving global) — **user-approved**: warm-started locals diverge far after long
training anyway, and we compete on *relative* metrics vs CLEAR, not on beating
each expert.

- **Architecture: Impala-CNN large** — policy kinds `impala_ac` (single-head
  expert) / `impala_ac_multihead` (global). Three conv sequences (32,64,64), each
  = conv3×3 → maxpool3×3/2 → 2 residual blocks, then FC 512. **4.36M params**
  (vs Nature-CNN 1.7M). `crl/policies/impala.py`.
- **10 games:** Pong, Breakout, Boxing, Freeway, SpaceInvaders, Qbert, Assault,
  Krull, Seaquest, **BeamRider** (newly added to `ATARI_GAMES`/`RANDOM_SCORES`).
- **Shared init** every expert (and later the global) forks from:
  `experts/_shared_init.pt` (seed 12345). Regenerate: `python -m
  experiments.make_shared_init`.
- **Trainer:** `experiments/train_expert.py` — standard PPO, **trains to plateau**
  (stops after 6 non-improving greedy-100 evals past a 5M-frame floor; 60M safety
  cap; keeps the **best** checkpoint). Identical recipe for all 10 (Adam 2.5e-4,
  n_envs=16, n_steps=128, clip 0.1, GAE 0.95, γ 0.99).
- **Storage (`experts/<Game>/`, gitignored binaries):** `best_model.pt`,
  `last_model.pt`, `checkpoints/`, `train_log.csv` (per-iter frames / agent-steps
  / episodes / wall / training-return / greedy-100 / losses), `meta.json`
  (**frames/episodes/iters-to-best** — for a matched CLEAR budget), `config.json`,
  `reward_curve.png`. → *any* reward curve is replottable from `train_log.csv`.
- **Published (tracked):** `python -m experiments.experts_summary` copies curves +
  a budget table to `diagnostics/experts/` (auto-pushed by a monitor as each
  converges).
- **Launch:** `sbatch --partition=dgx_aic,gpu,AI_Center_L40S
  scripts/hpc_expert.sbatch <Game> 0`. One game per GPU on dgx-1 (8) + node242 +
  node493. ~2–5 h/game, ~5 h wall.

**When you return:** check `experts/*/meta.json` (or `diagnostics/experts/README.md`
= budget table). A game is done when `meta.json` exists. If one crashed, resubmit
just that game. Sanity-check the reward curves before trusting an expert.

### Next steps (in order)

1. **Higher-budget joint run** — settle capacity vs undertraining (gates whether
   to invest in more/modular capacity). Cheap, decisive. (`method: joint`,
   `experiments/train_expert`-style; joint mode is in `crl/ppo_continual.py`.)
2. **Behavior-constrained method** (the theory fix, worth doing regardless):
   current-task BC (`global_bc_coef`, done) **plus past-task KL preservation**
   (re-derived replay-free), ideally as a **dual-controlled KL constraint**.
   Rigorous version: constrain occupancy / successor features.
3. **Consolidation harness on the fixed experts:** global inits from the task-1
   expert; `V_L` = expert value (add the **`β·V_L` fractional-reference** lever so
   μ need not chase an unreachable bar); BC target = expert policy. **Env must use
   `max_steps=0`.**
4. **Fair CLEAR comparison** on relative metrics (retention/BWT/accuracy after the
   full sequence) with a **frame budget matched from the experts' `meta.json`**.

---

## Method + code facts (stable)

- Constraint is a per-task, one-sided, **squared** hinge (eqs 7, 11): penalty only
  where the trained policy is *below* its frozen reference. Local carries one
  `λ_i` per past task; global carries a single `μ`. Uniform weights **`ω_i = 1/k`**
  (current task count). **Replay-free via env access** (fresh rollouts in old
  envs). `ε` in **squared-value units**.
- Two optimizer backends implement the same formulation: **REINFORCE** (tabular /
  MinAtar; theory double-check) and **PPO** (Atari; `trainer.kind: ppo`). Only the
  ∇V estimator differs (`docs/REINFORCE_to_PPO.md`). Updated objective doc:
  `docs/Objective_for_Continual_Reinforcement_Learning (4).pdf` (adds the
  monitor-feasibility / check-before-distillation notes that motivated the
  diagnostics; the algorithm is unchanged).
- **PPO methods** (`crl/ppo_continual.py`, dispatched on `ppo.method`):
  `constrained` (min-max; local is standard-PPO = the "local-free" variant),
  `finetune`, `clear` (replay+cloning baseline, `crl/ppo/clear.py`), `joint`
  (feasibility upper bound). Diagnostics knobs: `diagnostics`, `diag_every`,
  `global_probe_head_only` (freeze-trunk probe), `global_bc_coef` (behavioral
  cloning). `crl/ppo/trainer.py::optimize_batches` normalizes actor coefficients
  by their sum (prevents dual-μ from starving the shared critic — keep this).
- Multi-head actor-critic; the **shared trunk is where forgetting lives**. Reported
  metric = **raw greedy game score**.

## Infra / gotchas

- **Stable GPUs:** `dgx_aic` (dgx-1, 8×A100), `gpu` (node242, P100),
  `AI_Center_L40S` (node493, L40S). **V100 partitions preempt/requeue** (restart
  from task 1) — avoid for long runs.
- **Background monitors sometimes get killed.** All results are regenerable from
  `results/` + `experts/` via the plotting scripts (`experiments/{joint_plot,
  value_constraint_plot,probe_plots,diagnostics_plots,experts_summary}.py`). When a
  monitor dies, just rerun the relevant plot script and commit.
- **Commits:** author `ProtikNag <protiknag08@gmail.com>`; end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. `results/` and
  `experts/` binaries are gitignored; `reports/` and `diagnostics/` (figures +
  small text) are tracked.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Adding things (registry-driven)

- **Env family:** subclass `TaskFamily` in `crl/envs/`, register in
  `FAMILY_REGISTRY`; set `success_on_termination` correctly.
- **Policy / estimator / dual controller:** same pattern under `crl/policies/`,
  `crl/estimators/`, `crl/duals/`; register in the corresponding registry.
- Keep hyperparameters in `configs/` (no magic numbers in scripts); the config
  snapshot is logged with every run.
