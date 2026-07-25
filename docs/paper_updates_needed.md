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
