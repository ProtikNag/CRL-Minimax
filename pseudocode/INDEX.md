# Pseudocode Index -- CRL-Minimax

Three algorithms documented as high-level pseudocode. Source math:
docs/Objective_for_Continual_Reinforcement_Learning.pdf (eqs 1-46). All docs
mirror the code as implemented, not an idealized version.

## 01  Min-Max Local/Global Alternation  (Formulation A)

  File   : 01_minmax_alternation.md / .pdf
  Select : ppo.method = constrained
  Source : crl/ppo_continual.py  (PPOAlternationTrainer._constrained_task,
           _train_first_task, run)
           crl/ppo/trainer.py    (LocalTrainer.train, GlobalTrainer.train)
  Eqs    : 2, 5-12, 16-32
  Summary: Per task, a LOCAL PPO phase from theta^0=phi produces a frozen
           current-task floor V_k^L; a GLOBAL phase then maximizes the
           weighted past-task lead sum omega_i V_i under a one-sided squared
           shortfall keeping V_k^G >= V_k^L, enforced by a single dual mu.
           Only the actor is constrained; critic/GAE are standard PPO.

## 02  Stored-Expert Consolidation  (Formulation B)

  File   : 02_stored_expert_consolidation.md / .pdf
  Select : ppo.method = stored_expert
  Source : crl/ppo/stored_expert.py (StoredExpertTrainer.train, _global_value)
           crl/ppo_continual.py      (_stored_expert, _load_all_experts,
             _init_global_from_expert)
  Eqs    : 34-46
  Summary: Frozen per-task expert ceilings V*_i. NO min-max, NO dual mu, NO
           local phase. Global regresses toward every ceiling via the
           symmetric gap coefficient omega_i*2*max(0, V*_i - V_i^G), with V_i^G
           a FRESH Monte-Carlo estimate (never the critic head). Incremental
           consolidation over tasks 2..K with a retention row per phase.

## 03  Shared PPO Update Core

  File   : 03_ppo_update_core.md / .pdf
  Source : crl/ppo/trainer.py  (PPOTrainer.optimize_batches)
  Eqs    : realizes the policy-gradient weights of 22-29 (A) and 46 (B)
  Summary: The one reusable PPO optimizer. Per-stream actor_coeff * clipped
           surrogate + standard value MSE + entropy bonus; actor coeffs
           renormalized by their sum (avoids starving the shared critic when
           the dual coefficient is large); single global grad-norm clip. All
           continual-learning weighting enters only through actor_coeffs.

## Notes / possible code<->math divergences

  (see final report accompanying this index)
