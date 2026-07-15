# REINFORCE → PPO: exactly what changed in the update equations

PPO is used **only as the optimizer** that estimates the policy-gradient ∇V.
The continual-learning formulation — tasks, objectives, constraints, Lagrangians,
local/global alternation, dual updates — is **unchanged**. Equation numbers below
are those of `docs/Objective_for_Continual_Reinforcement_Learning.pdf`.

## The single quantity that changes: the estimator of ∇V

Every update in the paper is assembled from one primitive: the policy-gradient of
a task value, ∇_θ Vᵢ. Only its **estimator** changes.

- **REINFORCE (eqs 17–19, 21–23).** Score-function estimator
  ∇_θ Vᵢ ≈ 𝔼_τ[ Σₜ γᵗ ∇_θ log π_θ(aₜ|sₜ) · Âₜ ],  with Âₜ = reward-to-go Gₜ minus a
  baseline. Equivalently, the gradient of the surrogate
  L^PG(θ) = 𝔼[ Σₜ log π_θ(aₜ|sₜ) · Âₜ ].
- **PPO (this port).** Replace L^PG by the clipped surrogate
  L^CLIP(θ) = 𝔼[ min( rₜ(θ) Âₜ, clip(rₜ(θ), 1±ε_clip) Âₜ ) ],  rₜ(θ) = π_θ(aₜ|sₜ)/π_{θ_old}(aₜ|sₜ),
  with Âₜ from **GAE(λ)** using a learned critic V_ψ.

Near θ_old the two have the **same expected gradient** (∇L^CLIP ≈ ∇Vᵢ); PPO's is
lower-variance and trust-region-limited. So wherever the paper writes ∇Vᵢ, we now
substitute ∇L^CLIP on fresh rollouts of task i. Nothing else in the equations moves.

## Equation-by-equation

| Eq | Role | REINFORCE | PPO port |
|----|------|-----------|----------|
| 1  | MDP / task | — | unchanged |
| 2  | θ⁽⁰⁾ = φ (local starts from global) | — | **unchanged** |
| 7–9 | Local problem: max current-task lead s.t. per-past-task squared shortfall F_{L,i}=max(0,Vᵢ^G−Vᵢ^L)² ≤ εᵢ, multiplier λᵢ | — | **unchanged** (definitions, hinge, εᵢ, λᵢ) |
| 10 | Local Lagrangian / saddle | — | **unchanged** |
| 11–13 | Global problem: max past lead s.t. F_G=max(0,V_k^L−V_k^G)² ≤ ε, multiplier μ | — | **unchanged** |
| 14 | Global Lagrangian | — | **unchanged** |
| 17–19 | Estimator of ∇V | score-function (∇log π · Â) | **→ ∇L^CLIP (clipped surrogate + GAE)** |
| 20, 22 | Local primal step: θ ← θ + α[ ω_k ∇V_k + Σᵢ λᵢ·2·sfᵢ·∇Vᵢ ] | uses REINFORCE ∇V | **same formula**, ∇V := ∇L^CLIP |
| 24 | Local dual step: λᵢ ← [ λᵢ + η(F_{L,i} − εᵢ) ]₊ | — | **unchanged** |
| 26 | ∇F chain rule: ∇F_i = −2·max(0,·)·∇Vᵢ (coeff 2·shortfall) | — | **unchanged** (feeds the same coeff; ∇Vᵢ := ∇L^CLIP) |
| 30 | Global primal step: φ ← φ + β[ Σᵢ ωᵢ ∇Vᵢ + μ·2·sf·∇V_k ] | uses REINFORCE ∇V | **same formula**, ∇V := ∇L^CLIP |
| 32 | Global dual step: μ ← [ μ + η(F_G − ε) ]₊ | — | **unchanged** |

`sf` = shortfall = max(0, reference − value). ω uniform = 1/k (Setup).

## Three things that are new — but do not alter the formulation

1. **Critic + GAE (advantage estimation only).** A value net V_ψ per policy
   (local, global) trained by standard MSE to returns, used solely to form Âₜ in
   L^CLIP. It is **never** constrained, distilled, or regularized — only the actor
   carries the CL constraint. This changes *how* ∇V is estimated, not any equation.

2. **Constraint values stay Monte-Carlo.** The F-hat values Vᵢ that drive the
   shortfalls and dual updates (eqs 7, 11, 24, 32) are estimated from fresh
   episode **returns** (the paper's definition), not from the critic — so the
   critic's approximation error never enters the multiplier dynamics.

3. **Actor-coefficient normalization (implementation, not theory).** In eqs 20/30
   the bracket is a nonnegative-weighted sum of gradients Σⱼ cⱼ ∇Vⱼ (cⱼ ∈ {ωᵢ,
   λᵢ·2·sfᵢ, μ·2·sf}). We divide the bracket by Σⱼ cⱼ before the step. This is a
   **positive scalar rescaling of the ascent direction**, so it leaves the
   direction — and hence the primal–dual fixed point / KKT conditions — unchanged;
   it only bounds the step magnitude. It is required because PPO shares one trunk
   between actor and critic under global grad-norm clipping: an unbounded dual
   coefficient (μ→cap) would otherwise dominate the clipped gradient and starve
   the critic. With the normalization, the critic keeps learning and the global
   consolidates the current task up to its local reference.

## One-line summary

Wherever the paper takes a policy gradient ∇V (eqs 20, 30 via 17–19 and 26), we
compute it with PPO's clipped surrogate + GAE instead of the REINFORCE score
function. The constraints (7–14), the shortfall hinges, the weights, and the dual
ascent steps (24, 32) are byte-for-byte the same method.
