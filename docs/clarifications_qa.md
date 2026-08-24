# Clarifications Q&A log

Running record of clarifying questions asked during development and their concise
answers. Kept for paper writing and rebuttals — each entry is a point that was
non-obvious enough to need pinning down. Newest topics first within each section.

Reference convention throughout: the **local model** (per-task specialist) is the
Part-A reference, **not** stored experts. Reported eval is **greedy, 100 rollouts**.

---

## A. Our method vs CLEAR (the "are we reinventing CLEAR?" thread)

**Q: How does CLEAR actually work — does it store frames, and does a model generate the transitions?**
A: CLEAR (Rolnick et al. 2019) is **not** generative replay — it stores **real**
experienced transitions in a replay buffer: `(observation/frames, action, reward,
done, behavior-policy probs μ(·|s), value V(s))`, one big pool mixed across all
past tasks (no task boundaries). A single shared actor-critic network is **trained
on** this data (never generates it). Each batch mixes new + replayed experience
(~50/50) and applies three losses on the replayed half: (1) **V-trace off-policy
actor-critic** RL loss, (2) **policy cloning** `KL(stored μ ‖ current π)`, (3)
**value cloning** `‖V(s) − V_stored(s)‖²`. Losses 2–3 are the anti-forgetting
mechanism: pin the network to reproduce its past actions/values on past states.

**Q: Is our value-matching the same as CLEAR's value cloning (step 3)? Especially if
we precompute the local model's values on stored windowed states and then shrink the
gap between the global's value and those cached targets?**
A: It hinges on **one fork — do we make the global's *critic predict* the target, or
make the global's *policy achieve* it?**
- **Critic regresses to the cached scalar** → yes, that is CLEAR value cloning, up to
  three tweaks: *one-sided* (hinge; penalize only falling below, CLEAR is symmetric),
  *dual-weighted* (μ constraint vs CLEAR's fixed coefficient), *reference = dedicated
  frozen expert* (vs CLEAR's own past prediction).
- **Policy must achieve the return by acting** → **not** value cloning. Closing the gap
  is a policy-gradient step on **realized return** (`μ·2·shortfall·∇_φ V_φ`) that
  **pushes the actor** and needs **live rollouts of the current global** — it cannot
  come from a static cached target.
Our method does the second: CLEAR pushes the **critic** ("predict the number"), ours
pushes the **actor** ("earn the number"). Precomputing the *reference* is fine/cheap
and doesn't change our identity; but if we also switch to regressing the global's
*predicted* value, we've converted our performance-constraint into CLEAR value cloning
— and value-matching alone is insufficient (see the KL≈0.8 diagnostic below), so we'd
then need policy cloning too and would essentially be re-deriving CLEAR.

**Q: What is "ours + BC"?**
A: Our min-max method **plus a behavioral-cloning term** on past tasks: make the
global's action distribution match the local reference's on past-task states, i.e.
minimize `KL(reference π ‖ global π)` (the `global_bc_coef` knob). It is a *retention*
loss and is unrelated to the trust-region KL below.

**Q: What actually distinguishes us from CLEAR, then?**
A: The combination CLEAR does not have: a **realized-return value *constraint*
(actor-pushed)** + an **adaptive dual μ (min-max)** + **one-sidedness** (improvement
allowed). If we drop the dual/min-max and just do stored-data value+behavior matching,
we *are* basically CLEAR. So the paper's contribution must live in the primal-dual
constraint, **not** in "we match value and behavior."

**Q: How do we make a fair (apples-to-apples) comparison with CLEAR given we use
different "memories"?**
A: The two methods remember in different currencies — CLEAR = stored raw transitions,
no new past-env interaction; ours = frozen models + heavy live past-env rollouts
(O(k)/global-iter). No single knob equalizes them. Plan: **equalize total environment
frames** (the honest RL sample budget — count current-task learning + our past-env
rollouts + any env steps CLEAR uses), same net / task order / greedy-100 eval, and
**report the two storage footprints side by side** (CLEAR buffer size vs our k models)
rather than pretend they're equal. Run two comparisons: (1) headline — standard CLEAR
vs ours, matched frames; (2) mechanism ablation — hold data/access fixed and isolate
whether the dual value-constraint beats fixed-weight replay+cloning. (CLEAR loss
details above are from memory — verify against the paper before building the central
claim on them.)

---

## B. Evaluation & metrics (reading the forgetting matrices)

**Q: In fig2 (local-normalized), Qbert shows 1.07 after task 1 — but nothing else has
happened yet, so shouldn't it be exactly 1.0?**
A: Yes, 1.0 is correct. The 1.07 was a **measurement inconsistency** in the plotted
reference, not a real effect: the matrix cell was a greedy-100 eval of the best-selected
task-1 model (4468), but the reference had fallen back to a noisy **15-episode**
`final_greedy` (4172) from the resume source. Task 1 has no consolidation, so its local
reference IS its own greedy-100 diagonal → fixed to 4468 → cell = 1.000.

**Q: Just after learning SpaceInvaders we're getting 133% retention on it — is that
right?**
A: No — artifact of dividing by the wrong reference field. We used SI's 3-episode
`final_greedy` (575) instead of the greedy-100 `local_greedy` (1132). With the correct
reference, SI is **67.6%**, i.e. genuinely under-consolidated, not over-performing.
General rule: the reference must be the greedy-100 `local_greedy` (same measurement
basis as the diagonal), never the noisy stop-eval `final_greedy`/`best_greedy`.

**Q: Why does Boxing show >100% on its own diagonal (after task 4)?**
A: Benign and expected for a **current-task** cell. At its own task, Boxing is the
maximization target and the global trains on it for extra iters (local 1000 + global
2000) beyond the local's 1000-iter snapshot, so it can exceed the (noisy, still-rising)
local point estimate. Diagonal >100% just means "continued consolidation beat the local
snapshot." Only an **off-diagonal** (a *past* task) exceeding 100% is genuine backward
transfer and worth scrutiny.

---

## C. Method internals

**Q: We removed everything KL-related from global consolidation — so what was the
KL≈0.8 you mentioned?**
A: Two different KLs. (1) The **trust-region / early-stop KL** (`target_kl`) is a PPO
*optimization guardrail* limiting per-update step size — that is what we removed from
global (`kl_stop=False`). (2) `KL(local‖global) ≈ 0.8` was a **diagnostic
measurement** of behavioral divergence between the consolidated global and the local
specialist — never part of the loss, just a probe. They're unrelated; deleting the
trust-region touched nothing about behavior matching. The diagnostic showed value can
match while behavior diverges — motivating the optional BC term.

**Q: (Global objective) Is V_πG basically a constant so we're only maximizing V_πθ?**
A: For the **local** update the global is frozen, so `V_π̄G` is a constant offset and
the local step is effectively `∇_θ V_k^{π_θ}` (maximize current-task value) subject to
the past-task λ constraints. Symmetrically, the global update freezes the local and
maximizes `Σ_{i<k} V_i^{π_φ}` subject to the current-task μ constraint. The frozen
reference terms drop out of the gradient; only the trained policy's value gradient
and the constraint terms remain.

---

## Open items flagged during these discussions (not yet resolved)
- **Better retention metric** — current metric penalizes "competent but sub-specialist"
  play (e.g. Breakout 0.38: well above random, but tiny vs an extremely strong local).
  Need a metric that credits still-playable-but-not-expert performance.
- **Global KL trust-region** — whether to re-add a per-update trust region (stabilizer,
  distinct from the rejected task-level early-stop) to stop the Boxing⟂SI consolidation
  interference. Awaiting sign-off.
- **Verify CLEAR loss set against the primary paper** before the "why we differ"
  argument goes into the writeup.
