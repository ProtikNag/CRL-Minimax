# Diagnostics index

Each subfolder holds the figures + verdict for one experiment probing *why* the
constrained min-max method under-consolidates / forgets. All reported scores are
**greedy, 100 rollouts** (project rule — no stochastic eval). Values `V_G`/`V_L`
in the traces are the on-policy (stochastic) constraint values by definition.

| Folder | Experiment | Question it answers |
|---|---|---|
| `atari_diag_<seed>/` | **Core diagnostics** of the constrained run (order: Pong→Breakout→Boxing→SpaceInvaders). 8-panel per global phase: V_L/V_G/gap, μ, F_G vs ε, ‖g_new‖/‖g_old‖, cos, current-task greedy trajectory, per-old-task value; plus the greedy retention matrix. | Does μ pin? Does the constraint stay active? Does the current-task gradient dominate? Is retention held (greedy)? |
| `gradient_conflict/<run>/` | **Per-task gradient alignment.** cos(g_new, g_i) for *every* old task vs the aggregate cos(g_new, Σ g_i). | Is the aggregate "orthogonality" real, or does it hide per-task alignment/conflict that cancels in the sum (e.g. Pong +0.8, Breakout −0.7)? |
| `head_only_probe/<run>/` | **Head-only consolidation probe** (2 task orders). Global phase inits from the local, freezes the trunk, consolidates only the heads. `current_task_VG_vs_VL` + retention matrix. | Is the damage from shared-representation (trunk) updates, or from the constrained objective itself compromising the current task? |
| `feasibility/<run>/` | **Experiment 1 — feasibility upper bound.** Joint model (all games at once, no constraint) vs single-task ceiling, at equal per-game budget. | Does a single shared θ that does all tasks well *exist*? (joint ≥ ceiling ⇒ feasible ⇒ the problem is objective/optimization, not capacity/infeasibility.) |
| `value_constraint/<run>/` | **Experiment 2 — value-constraint sufficiency.** Per phase: current-task value gap (V_L−V_G) vs behavioral gap KL(π_local‖π_global). Plus the BC-intervention arm (`global_bc_coef>0`). | Does the value gap → 0 while KL stays large? (⇒ the scalar value constraint is satisfied by a behaviorally-different policy — too weak.) Does adding behavioral cloning fix consolidation? |

## Experiment 2 arms
- `exp2a_klmeasure_seed0` — BC **off** (pure value constraint): the measurement.
- `exp2b_bc_seed0` — BC **on** (`global_bc_coef=0.1`): the intervention. Compare retention + KL against 2a.

## Regenerating
```
python -m experiments.diagnostics_plots   --run results/atari_diag_seed0        # core + gradient_conflict
python -m experiments.probe_plots         --run results/atari_probe_orderA_seed0 # head_only_probe
python -m experiments.joint_plot          --run results/exp1_joint_seed0         # feasibility
python -m experiments.value_constraint_plot --run results/exp2a_klmeasure_seed0  # value_constraint
```
