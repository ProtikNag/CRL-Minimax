# Pseudocode Index -- CRL-Minimax

Pseudocode docs for the implemented algorithms. All docs mirror the code as
implemented, not an idealized version.

Source math: `docs/Updated_Objective_for_CRL.pdf` (Part A, eqs 1-48). This
branch (`feature/updated-objective`) implements **Part A only** (min-max,
experts NOT stored). Part B ("When we have the expert models stored") is
deferred and has been stripped from the code, so its pseudocode doc (formerly
02) was removed; the former doc 01 cited the old objective PDF and a removed
shortfall path and was superseded by doc 04 and removed.

## 04  Part A -- Min-Max Consolidation, experts NOT stored  (CURRENT)

  File   : 04_partA_minmax_no_stored_experts.md / .pdf
  Select : ppo.method = constrained
  Source : crl/ppo_continual.py  (PPOAlternationTrainer.run, _train_first_task,
             _constrained_task, _evaluate_row)
           crl/ppo/trainer.py    (LocalTrainer.train, GlobalTrainer.train)
           crl/duals/controllers.py (ProjectedAscentDual.update)
  Math   : docs/Updated_Objective_for_CRL.pdf, Part A, eqs 1-48.
  Eqs    : 4-8, 18-23, 25-29, 38, 40, 42, 47 (mapped explicitly inside the doc).
  Status : Implementation PASSED code-verifier against eqs 26/38/40/47; the
           underlying Part A derivation is math-verifier SOUND.
  Summary: Three-checkpoint cycle pi_G^- -> pi_L -> pi_G^+. Stage 1 = plain PPO
           on the global net. Stage k>=2: a LOCAL PPO phase from theta^0=phi^-
           produces the frozen current-task floor V_k^L; a GLOBAL phase then
           maximizes the weighted past-task objective sum_{i<k} omega_i V_i^phi
           (omega_i=1/k) as a primal-dual saddle, with the current task entering
           through the one-sided squared hinge F_k = [V_k^L - V_k^G]_+^2 <= eps
           (coeff_k = 2 mu [V_k^L - V_k^G]_+) and dual mu <- [mu+eta(F_k-eps)]_+.
           Replay-free fresh past+current rollouts each iter; V_k^{L,G} are
           Monte-Carlo (never the moving critic head); only the actor constrained.
           Two faithful deviations: coeff-sum normalization (direction-preserving)
           and a mu ceiling (both magnitude-only, saddle point unchanged).
           Post-task greedy eval (forgetting) matrix.

## 03  Shared PPO Update Core

  File   : 03_ppo_update_core.md / .pdf
  Source : crl/ppo/trainer.py  (PPOTrainer.optimize_batches)
  Summary: The one reusable PPO optimizer shared by the local and global phases.
           Per-stream actor_coeff * clipped surrogate + standard value MSE +
           entropy bonus; actor coeffs renormalized by their sum (avoids starving
           the shared critic when the dual coefficient is large); single global
           grad-norm clip. All continual-learning weighting enters only through
           actor_coeffs.
