# Changes to reflect in the objective PDF / paper

Design decisions made during the Atari experiments that **deviate from
`docs/Objective_for_Continual_Reinforcement_Learning (4).pdf`** and must be
written into the paper/derivation. Listed newest-first.

## 1. RELATIVE (normalized) constraint  ⚠️ update the constraint + derivation

The current-task constraint changed from the **absolute** shortfall
`F_G = (V_k^L − V_k^G)²`, enforce `V_k^G ≥ V_k^L` (eqs 9–11, 23–33), to a
**relative** one:

- shortfall `s = max(0, (V_k^L − V_k^G) / |V_k^L|)`  (a fraction in [0,1])
- `F_G = s²`, constraint `s ≤ δ`, i.e. **`V_k^G / V_k^L ≥ 1 − δ`**
- tolerance `ε = δ²` in squared-relative units; we use **δ = 5% → ε = 0.0025**
- primal coefficient `coeff_k = μ · 2 · s` (relative shortfall)

**Why:** the (discounted, sign-clipped) value scale differs a lot across games
(e.g. Pong V≈0.03 vs Breakout V≈2.5), so an absolute `ε` is a tight bar on some
games and trivially loose on others (the constraint is effectively *off* where
the whole value is below `ε`). Normalizing by `V_k^L` makes "reach the expert"
mean the same fraction on every task. Update eqs 9–11 and the `∇_φ F_G` /
`∇_φ L_G` derivation (23–33) accordingly (the `1/|V_k^L|` factor; in code it is
absorbed by PPO's per-minibatch advantage normalization, which makes the
current-task PG direction unit-scale — worth a sentence).

Config: `ppo.constraint_relative`, `trainer.eps`.

## 2. Fixed expert references instead of `θ⁰ = φ` (eq 2)

Local models are **precomputed single-task experts** (trained once, unconstrained,
from a shared init) and reused as **fixed** references — the local phase is
dropped. So `V_k^L` is a fixed expert value, not a warm-started local. The global
inits from the task-1 expert. Document as the "fixed-expert reference" variant
(replaces eq 2). We compete on *relative* metrics vs CLEAR, not on matching each
expert.

## 3. Value estimated by deterministic no-op enumeration

Greedy on `repeat_action_probability=0` Atari is deterministic given the start,
so `V` is estimated **exactly** by enumerating the `noop_max` random-start
no-op counts (one rollout each), not by Monte-Carlo sampling. Worth a line in the
evaluation/estimator section (replaces the `1/N Σ G(τ)` MC estimator for greedy V).

## 4. No per-episode cap

Atari envs use `max_steps=0` (no `TimeLimit`); episodes end only on true
termination, so `done = terminated` and GAE bootstrapping is exact. (A `TimeLimit`
+ `done=term|trunc` biases the value on long episodes.)

## 5. Constraint form: floored additive hinge (not a ratio)

The relative constraint `V_G/V_L ≥ 1−δ` / normalized shortfall carries `1/|V_L|`
in the gradient and blows up for near-zero-value tasks (Pong `V_L=0.031` → 34×).
We use the **floored additive hinge** `loss = (max(0, (V_L−V_G) − δ·max(|V_L|,τ)))²`
— same feasibility boundary, gradient in raw value units (numerically stable).

## 6. Retention must constrain ALL tasks (past + current), by relative shortfall

Key correction to the min-max realization: constraining only the *current* task
(fixed weight ω on past tasks) breaks retention two ways — past tasks get no
adaptive shortfall/mid-late protection, and an unbounded current-task coefficient
`μ·2·shortfall` (normalized against ω) starves the past tasks to ~0 gradient
(measured: `coeff_k=18,750` → past task 0.003% of the update). The fix is a true
min-max: per-task floored shortfall (incl. intermediate mid/late states) for every
task, weighted by the **relative** shortfall `w_i = shortfall_i/Σ_j shortfall_j`
(bounded, sums to 1) so the worst-retained task is pushed hardest without starving
the others. See F16 in atari_findings.md.

## 7. Mid/late-game retention + intermediate-state constraint

Retention failure is concentrated in the **mid/late game** (Phase B: Breakout
global/expert return decays 0.63→0.58→0.26 early→mid→late). The 50 no-op-*start*
evaluation only samples the opening and misses this. We add N mid/late states
(sampled from expert trajectories, reached by exact deterministic re-simulation)
to the constraint, with a per-state floored shortfall.
