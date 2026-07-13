# Constrained Two-Policy Continual Reinforcement Learning

A two-policy continual RL framework with constrained min-max updates. A
**local policy** learns the current task while a **global policy** consolidates
the past; the two are trained in alternation by primal-dual **policy gradient
(REINFORCE)**. The past is protected by a per-task, one-sided, *squared* value
constraint, not a replay penalty.

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
values `J_k(π) = Σ_i ω_i V_i^π` with **uniform weights `ω_i = 1/k`** (where `k`
is the number of tasks seen so far, so the per-task weight shrinks as the
sequence grows). The improvement `J_k(π) − J_k(π′)` splits into a current-task
term and a past-task term (derivation eq 6), motivating one model per term.

**Replay-free via environment access.** Past-task values are estimated from
*fresh rollouts in the old environments*, not from a stored transition buffer.
The method keeps no replay buffer; it assumes past environments remain
instantiable (true for the MinAtar family here).

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

### Variant: unconstrained local

A studied variant drops the local phase's `λ` constraint
(`trainer.local_unconstrained: true`) so the local policy is a pure-plasticity
learner of the new task; only the global phase's `μ` constraint remains. It
tests whether fully mastering the new task locally, then consolidating, helps
retention. Both the full theory and this variant are compared below.

## 3. Repository layout

```
crl/
├── config.py            # typed YAML config, strict unknown-key rejection
├── seeding.py           # set_seed (torch/numpy/random/cuda)
├── envs/                # minatar (the experiment family) + gridworld (tabular,
│                        #   kept only as the exact-gradient TEST HARNESS)
├── policies/            # cnn / cnn_multihead (MinAtar); mlp, tabular (harness)
├── estimators/          # monte_carlo (REINFORCE); exact DP (harness only)
├── duals/               # projected-ascent and PID controllers
├── evaluation.py        # rollout performance metrics (env-aware success)
├── baselines.py         # sequential fine-tune + joint multi-task baselines
├── trainer.py           # the alternation loop (eqs 22-24 / 30-32)
experiments/  run.py · multiseed_comparison.py (one seed -> N methods) ·
              aggregate_theory.py (3-method MinAtar CI bundle) ·
              aggregate_seeds.py (generic CI bundle) · sweep.py
analysis/     plots.py (single-run diagnostics) · aggregate.py (multi-seed CI
              figures/tables) · style.py
configs/      minatar_multihead (constrained-local, full theory) ·
              minatar_localfree (unconstrained-local variant) ·
              gridworld_exact / gridworld_sampled (test-harness fixtures)
tests/        gradient checks · estimator agreement · dual dynamics · end-to-end
              (all on the exact-gradient gridworld harness — the theory check)
```

Everything is registry-driven: add an env family, policy, estimator, or dual
controller by registering it in the relevant `__init__.py`; configs select by
name. No component knows another's internals.

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                             # 25 tests, ~1.5 min

# One seed of the MinAtar study (constrained-local + fine-tune):
python -m experiments.multiseed_comparison \
    --config configs/minatar_multihead.yaml --name minatar_multihead \
    --seed 0 --methods constrained finetune
# Unconstrained-local variant:
python -m experiments.multiseed_comparison \
    --config configs/minatar_localfree.yaml --name minatar_localfree \
    --seed 0 --methods constrained
```

The full study is 5 seeds × 3 method-runs on SLURM
(`scripts/hpc_minatar.sbatch <method> <seed> <config> <name>`), aggregated by
`experiments/aggregate_theory.py`:

```bash
python -m experiments.aggregate_theory --seeds 0 1 2 3 4
# -> reports/minatar_theory/figures/{png,svg}/ and tables/*.csv
```

**Headline experiment.** Four MinAtar games (SpaceInvaders → Breakout → Asterix →
Seaquest) learned in sequence with a shared conv trunk + per-task action heads,
pure REINFORCE, comparing three procedures on identical tasks/network/estimator:

- **Constrained-local min-max (ours, full theory)** — local constrained on past
  tasks (`λ`), global constrained on current (`μ`).
- **Unconstrained-local min-max (ours, variant)** — local free, global
  constrained.
- **Naive fine-tuning** — one net trained on each game in order (forgets).

Reported metric is the **raw game score** (undiscounted return), normalized
per-task for training only (`reward_scale = 1/expert`, so returns are ~O(1) and
the single squared-value `ε` is balanced across games). Results table and
figures land in `reports/minatar_theory/` — see the bundle for per-task
retention curves, learning/reward curves, forgetting matrices, and score tables
(mean ± 95% CI over 5 seeds).

**Result** (final policy, raw game score, 100 eval episodes; 5 seeds for
unconstrained-local, 4 for the others — one constrained/finetune seed still
running). Ours **retains the old games** where naive fine-tuning catastrophically
forgets them; fine-tuning only leads on the two most-recent games it has not yet
forgotten:

| Method | SpaceInv | Breakout | Asterix | Seaquest | Mean |
|--------|---------:|---------:|--------:|---------:|-----:|
| Random | 2.8 | 0.5 | 0.5 | 0.1 | 1.0 |
| **Constrained-local min-max (ours)** | **33.6** | **2.96** | 0.98 | 1.30 | **9.7** |
| **Unconstrained-local min-max (ours)** | **35.6** | 2.22 | 0.62 | 1.52 | **10.0** |
| Naive fine-tuning | 3.8 | 0.34 | 3.19 | 4.79 | 3.0 |

Fine-tuning collapses the two oldest games to ≈ random (SpaceInvaders 33→3.8,
Breakout →0.34) while acing the two it trained last; both min-max variants hold a
balanced profile across all four, retaining SpaceInvaders (~9×) and Breakout (~9×)
that fine-tuning erases. The cost is reduced plasticity on the newest games
(Asterix/Seaquest), where the constrained global under-learns relative to a fully
plastic single net — the honest retention-vs-plasticity trade. Per-game panels,
learning/reward curves, forgetting matrices and CI tables:
`reports/minatar_theory/`.


## 5. Benchmark tiers

1. **Exact gridworld harness** (`gridworld_exact`, `tests/`). Zero-variance DP
   estimator; verifies the update rules exactly (the constrained trainer prevents
   forgetting; the sampled REINFORCE gradient matches the exact policy gradient).
   Not an experiment — the theory double-check.
2. **MinAtar (headline).** Four games, shared conv trunk + per-task heads,
   REINFORCE, 5 seeds. Genuine cross-game interference in the shared trunk —
   real forgetting to prevent, unlike compatible toy tasks. This is the paper
   tier for the current write-up.
3. **Future.** Larger MinAtar / ALE subsets; value-based learners were explored
   but set aside to stay within the policy-gradient theory.

## 6. Open issues (ordered by severity)

1. **Alternation stability.** Each phase moves the other's frozen reference;
   the pair can cycle. Gap sequences are always logged (`phase: gaps`) to
   detect it. No general convergence theory — this is the research contribution.
2. **Feasibility under conflict.** Tight `ε` may be infeasible when tasks
   demand different actions in shared states; `λ`/`μ` then saturate. The
   multi-head policy (shared trunk + per-task heads) addresses the output side.
3. **REINFORCE sample efficiency.** MinAtar games are only moderately learnable
   under pure REINFORCE within a tractable budget; the budget is raised over the
   first proof-of-concept so each new game has time to learn. Fresh past-task
   rollouts scale with `k`; mitigated by once-per-phase frozen references,
   `past_task_sampling: sample` (unbiased O(1) past term), and batched
   `vector_rollout` per env.

## 7. Metrics

Continual World conventions: average performance, forgetting, forward
transfer, plus the full forgetting matrix (task × training-phase). Fixed
evaluation seeds; evaluation episodes separate from training rollouts. Figures
report task performance (game score), never discounted value.

## 8. Key references

Verified against primary sources; see `docs/citation_corrections.md` for the
full audit of the literature summary.

- Achiam et al., *Constrained Policy Optimization*, ICML 2017 — frozen-reference constraint surrogate.
- Rolnick et al., *Experience Replay for Continual Learning* (CLEAR), NeurIPS 2019 — replay + V-trace + behavioral anchor.
- Tessler et al., *Reward Constrained Policy Optimization*, ICLR 2019 — Lagrangian policy gradient.
- Stooke et al., *Responsive Safety in RL by PID Lagrangian Methods*, ICML 2020 — dual controller variant.
- Young & Tian, *MinAtar*, 2019 — miniaturized Atari benchmark.
- Wołczyk et al., *Continual World*, NeurIPS 2021 — benchmark + metrics.
- Kakade & Langford, 2002 — performance difference lemma.
```
