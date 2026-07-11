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
├── envs/                # gridworld (tabular) + cartpole; both expose a fast
│                        #   batched vector_rollout used by the MC estimator
├── policies/            # tabular, MLP, multihead (shared trunk + per-task heads)
├── estimators/          # exact DP, monte_carlo (REINFORCE), surrogate (stub)
├── duals/               # projected-ascent and PID controllers
├── evaluation.py        # rollout performance metrics (env-aware success)
├── baselines.py         # sequential fine-tune + joint multi-task baselines
├── trainer.py           # the alternation loop (eqs 22-24 / 30-32)
experiments/  run.py · multiseed_comparison.py (one seed -> N methods) ·
              aggregate_seeds.py (mean ± 95% CI bundle) · sweep.py
analysis/     plots.py (single-run diagnostics) · aggregate.py (multi-seed CI
              figures/tables) · compare.py (single-seed) · style.py
configs/      gridworld_tentask_sampled (headline) · gridworld_nn_three_task ·
              gridworld_exact · cartpole_multihead · ...
tests/        gradient checks · estimator agreement · dual dynamics · end-to-end
```

Everything is registry-driven: add an env family, policy, estimator, or dual
controller by registering it in the relevant `__init__.py`; configs select by
name. No component knows another's internals.

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                             # 25 tests, ~1.5 min

# Headline: 10-task, 5-seed, REAL-ROLLOUT proof-of-concept (one seed shown here;
# the full study is a SLURM array -- see scripts/hpc_tentask.sbatch):
python -m experiments.multiseed_comparison \
    --config configs/gridworld_tentask_sampled.yaml --name gridworld_tentask --seed 0
python -m experiments.aggregate_seeds \
    --config configs/gridworld_tentask_sampled.yaml --name gridworld_tentask --seeds 0
# -> reports/gridworld_tentask/figures/{png,svg}/  and  tables/*.csv
```

**Headline result** (10-task 9×9 gridworld, multi-head MLP, **sampled/Monte-Carlo
estimator — real rollouts**, 5 seeds). Mean over all 10 tasks of the deployed
policy's final value (±std) and task success rate (fraction of episodes reaching
the goal):

| Method | Mean value (±std) | Mean success | Fails on |
|--------|-------------------|--------------|----------|
| **Constrained min-max (ours)** | **0.848 ± 0.027** | **99.6%** | nothing (retains all 10) |
| Joint multi-task (upper bound) | 0.876 ± 0.003 | 100% | — (ceiling) |
| Unconstrained ablation (duals off) | 0.759 ± 0.039 | 95.0% | newest task (T10 → 0.27) |
| Naive fine-tuning (single net, sequential) | 0.542 ± 0.093 | 78.6% | oldest tasks |

The two standard baselines fail in *opposite* directions: naive sequential
fine-tuning forgets the **oldest** tasks (classic catastrophic forgetting), while
the constraint-off ablation forgets the **newest** (the global over-consolidates
the past). Our method retains all ten and matches the joint upper bound within
noise — now demonstrated at 10 tasks under real rollouts with error bars, not
just the earlier 3-task exact-estimator demo. A value ≈0.85 is optimal here
(~100% success, near-shortest paths). Value and success-rate tables (mean ± 95%
CI) land in `reports/gridworld_tentask/tables/`. The tabular 2-task
`gridworld_exact` remains the smallest zero-neural-network sanity demo.

Reproduce the full multi-seed study on a cluster, then aggregate:

```bash
sbatch scripts/hpc_tentask.sbatch          # array over seeds 0-4, all four methods
python -m experiments.aggregate_seeds \
    --config configs/gridworld_tentask_sampled.yaml \
    --name gridworld_tentask --seeds 0 1 2 3 4
```

Figures are written to `reports/<name>/figures/png/` and `.../svg/`, with
per-method diagnostics under `.../figures/<method>/`. The bundle: per-task
retention curves + summary with 95% CI bands, retention/performance tables and
a per-task performance bar chart, average-performance-vs-task curve, seed-averaged
forgetting matrix, plus per-method diagnostics (dual dynamics, gap sequences).

## 5. Benchmark tiers

1. **Exact gridworld** (`gridworld_exact`, `gridworld_nn_three_task`).
   Zero-variance DP estimator; verifies the update rules and, with the
   multi-head network, the multi-task retention result. The canonical demos.
2. **Many-task exact** (`gridworld_manytask_exact`). Six tasks on a 7×7 grid;
   zero-variance scaling check with the DP estimator. A fast sanity tier that
   isolates the formulation from sampling noise before the sampled run below.
3. **Sampled / many-task** (`gridworld_tentask_sampled`). REINFORCE estimator;
   real rollouts and sampling noise. **Done and validated:** the 10-task 5-seed
   headline above (ours retains all ten, both baselines forget in opposite
   directions). Runs as a SLURM array (`scripts/hpc_tentask.sbatch`); the fast
   vectorized gridworld rollout keeps each seed to ~15-20 min on CPU.
4. **CartPole family** (`cartpole_multihead`). Six physics-shifted tasks;
   first continuous-control tier (multi-head policy, sampled estimator, batched
   torch rollout). In progress.
5. **Paper tier.** MiniGrid goal families, then an Atari 3–6 game subset
   (Pong → Boxing → …), matching the scale of RePR. Requires actor-critic.

## 6. Open issues (ordered by severity)

1. **Alternation stability.** Each phase moves the other's frozen reference;
   the pair can cycle. Gap sequences are always logged (`phase: gaps`) to
   detect it. No general convergence theory — this is the research contribution.
2. **Feasibility under conflict.** Tight `ε` may be infeasible when tasks
   demand different actions in shared states; `λ`/`μ` then saturate. The
   multi-head + task-conditioned policy addresses this (used everywhere now).
3. **Rollout cost.** Fresh past-task rollouts every step scale with `k`.
   Mitigations in-repo: once-per-phase frozen references, `past_task_sampling:
   sample` (unbiased O(1) past term), and batched `vector_rollout` per env.

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
