# Constrained Two-Policy Continual Reinforcement Learning

A two-policy continual RL framework with constrained min-max updates. A
**local policy** learns the current task while a **global policy** consolidates
the past; the two are trained in alternation by primal-dual policy gradient.
The past is protected by a per-task, one-sided, *squared* value constraint,
not a replay penalty.

> **Note for Claude Code / new sessions.** Read this file and `HANDOFF.md`
> before touching code. The math is fixed in
> `docs/Objective_for_Continual_Reinforcement_Learning.pdf`; code comments cite
> its equation numbers. Implementation choices in this README are current and
> may evolve.

---

## 1. Problem setting

Tasks `1..k` arrive in sequence. Task `i` is an MDP
`M_i = (S, A, P_i, r_i, ρ_i, γ)` sharing state and action spaces; only
dynamics and reward vary. A policy is scored by a weighted sum of per-task
values `J_k(π) = Σ_i ω_i V_i^π` with **uniform weights `ω_i = 1/k`**. The
improvement `J_k(π) − J_k(π′)` splits into a current-task term and a past-task
term (derivation eq 6), motivating one model per term.

**Replay-free via environment access.** Past-task values are estimated from
*fresh rollouts in the old environments*, not from a stored transition buffer.
The method keeps no replay buffer; it assumes past environments remain
instantiable (true for the gridworld/CartPole/Atari families here).

## 2. Method

| Model  | Symbol      | Maximizes                | Constrained to stay near |
|--------|-------------|--------------------------|--------------------------|
| Local  | `π_L = π_θ` | current-task lead over G | global, on each past task |
| Global | `π_G = π_φ` | past lead over local     | local, on the current task |

Trained in **alternation**, each freezing the other; every local phase starts
from the global, `θ⁽⁰⁾ = φ` (eq 2).

The constraint is a **per-task one-sided squared shortfall** (eqs 7, 11): a
policy is penalized only where it falls *below* its frozen reference on a task
(that is what forgetting is), and not at all when it is at or above it.

**Local problem** (global frozen at `π̄_G`), one multiplier `λ_i` per past task:

```
max_θ  ω_k (V_k^{π_θ} − V_k^{π̄_G})
s.t.   F_{L,i}(π_θ) = max(0, V_i^{π̄_G} − V_i^{π_θ})² ≤ ε_i ,   i = 1..k−1
```

**Global problem** (local frozen at `π̄_L`), single multiplier `μ`:

```
max_φ  Σ_{i<k} ω_i (V_i^{π_φ} − V_i^{π̄_L})
s.t.   F_G(π_φ) = max(0, V_k^{π̄_L} − V_k^{π_φ})² ≤ ε
```

Differentiating the squared shortfall turns each constraint's contribution to
the primal step into a **scalar coefficient `2·shortfall`** times the ordinary
value gradient (eqs 18, 26). Primal updates (eqs 22, 30):

```
θ⁽ᵐ⁺¹⁾ = θ⁽ᵐ⁾ + α [ ω_k ∇θV̂_k + Σ_{i<k} λ_i · 2·max(0, V̂_i^G − V̂_i^θ) ∇θV̂_i ]
φ⁽ᵐ⁺¹⁾ = φ⁽ᵐ⁾ + β [ Σ_{i<k} ω_i ∇φV̂_i + μ · 2·max(0, V̂_k^L − V̂_k^φ) ∇φV̂_k ]
```

Dual updates are projected ascent on the squared-shortfall violation
(eqs 23, 31); a PID variant (Stooke et al. 2020) is available.

> **`ε` is in squared-value units.** A tolerated value gap `g` corresponds to
> `ε = g²`. A scalar `ε` is broadcast to every past task; a list sets `ε_i`.

## 3. Repository layout

```
crl/
├── config.py            # typed YAML config, strict unknown-key rejection
├── seeding.py           # set_seed (torch/numpy/random/cuda)
├── envs/                # task families: gridworld (tabular), cartpole
├── policies/            # tabular, MLP, multihead (shared trunk + per-task heads)
├── estimators/          # exact DP, monte_carlo (REINFORCE), surrogate (stub)
├── duals/               # projected-ascent and PID controllers
├── buffers.py           # per-task trajectory store (behavior log-probs)
├── logging_utils.py     # JSONL + config snapshot per run
└── trainer.py           # the alternation loop (eqs 22-24 / 30-32)
experiments/  run.py · sweep.py (grid, cluster-array ready) · compare_constraint.py
analysis/     plots.py (single-run) · compare.py (Fig 3 / Tab 1) · schematics.py
configs/      gridworld_exact · gridworld_sampled · gridworld_three_task ·
              gridworld_nn_three_task[_sampled] · cartpole_family
tests/        gradient checks · estimator agreement · dual dynamics · end-to-end
```

Everything is registry-driven: add an env family, policy, estimator, or dual
controller by registering it in the relevant `__init__.py`; configs select by
name. No component knows another's internals.

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                             # 21 tests, ~20 s

# Neural, 3-task headline result (constrained vs baseline + full figure bundle):
python -m experiments.compare_constraint \
    --config configs/gridworld_nn_three_task.yaml --name nn_three_task
# -> reports/nn_three_task/figures/{png,svg}/  and  tables/retention_table.csv
```

**Headline result** (3-task gridworld, multi-head MLP, exact estimator). Value
is what the algorithm optimizes; task performance is what matters. Both are
reported (`reports/<name>/tables/`):

| | Constrained (ours) | Unconstrained baseline |
|---|---|---|
| Value, per task | 0.83 / 0.83 / 0.83 | 0.83 / 0.83 / **0.20** |
| Success rate (goal reached) | 100% / 100% / **100%** | 100% / 100% / **69%** |
| Mean steps to goal | 4.8 / 4.5 / 4.4 | 4.9 / 4.5 / **59** |

A value of 0.83 is optimal here (100% success, ~4.5-step paths). The baseline
forgets the newest task: its deployed policy solves task 3 only 69% of the time
and wanders (~59 steps). The tabular 2-task `gridworld_exact` is the smaller
zero-neural-network sanity demo.

Figures are written to `reports/<name>/figures/png/` and `.../svg/`, with
per-method diagnostics under `.../figures/<method>/`. The bundle includes the
paper's must-have set: design-space map (Fig 1), method schematic (Fig 2),
per-task retention curves + summary (Fig 3), retention table (Tab 1), plus
diagnostics (dual dynamics, gap sequences, forgetting matrix).

## 5. Benchmark tiers

1. **Exact gridworld** (`gridworld_exact`, `gridworld_nn_three_task`).
   Zero-variance DP estimator; verifies the update rules and, with the
   multi-head network, the multi-task retention result. The canonical demos.
2. **Sampled gridworld** (`gridworld_sampled`,
   `gridworld_nn_three_task_sampled`). REINFORCE estimator; first test with
   real rollouts and sampling noise. **The recommended next experiment.**
3. **CartPole family** (`cartpole_family`). First continuous-control tier.
4. **Paper tier.** MiniGrid goal families, then an Atari 3–6 game subset
   (Pong → Boxing → …), matching the scale of RePR. Requires actor-critic.

## 6. Open issues (ordered by severity)

1. **Warm-start saturation** *(resolved by task heads for now).* With a *shared*
   (non task-conditioned) policy, `θ⁽⁰⁾ = φ` inherits a saturated softmax and
   later tasks fail to learn (`gridworld_three_task` still exposes this with a
   tabular policy). The **multi-head** policy (`policy.kind: multihead`, shared
   trunk + one head per task) removes it: the 3-task neural result learns and
   retains all tasks. A larger local learning rate is a cheaper partial
   mitigation (recovers the tabular case but is brittle under sampling noise).
   Revisit if a setting appears where per-task heads are not available.
2. **Alternation stability.** Each phase moves the other's frozen reference;
   the pair can cycle. Gap sequences are always logged (`phase: gaps`) to
   detect it. No general convergence theory — this is the research contribution.
3. **Rollout cost.** Fresh past-task rollouts every step scale with `k`.
   Mitigations in-repo: once-per-phase frozen references and
   `past_task_sampling: sample` (unbiased O(1) past term). See `HANDOFF.md`.
4. **Feasibility under conflict.** Tight `ε` may be infeasible when tasks
   demand different actions in shared states; `λ` then diverges. The
   task-conditioned policy option addresses this.

## 7. Metrics

Continual World conventions: average performance, forgetting, forward
transfer, plus the full forgetting matrix (task × training-phase). Fixed
evaluation seeds; evaluation episodes separate from training rollouts.

## 8. Key references

Verified against primary sources; see `docs/citation_corrections.md` for the
full audit of the literature summary.

- Achiam et al., *Constrained Policy Optimization*, ICML 2017 — frozen-reference constraint surrogate.
- Rolnick et al., *Experience Replay for Continual Learning* (CLEAR), NeurIPS 2019 — replay + V-trace + behavioral anchor.
- Tessler et al., *Reward Constrained Policy Optimization*, ICLR 2019 — Lagrangian policy gradient.
- Stooke et al., *Responsive Safety in RL by PID Lagrangian Methods*, ICML 2020 — dual controller variant.
- Kaplanis et al., *Policy Consolidation for Continual RL*, ICML 2019 — KL-cascade consolidation.
- Schwarz et al., *Progress & Compress*, ICML 2018 — distillation + online EWC.
- Wołczyk et al., *Continual World*, NeurIPS 2021 — benchmark + metrics.
- Kakade & Langford, 2002 — performance difference lemma.
