# Handoff

State of the project and what to do next. Written for a fresh session (on
Hyperion or elsewhere) that needs to continue without the prior context.

## Where things stand

The full method from `docs/Objective_for_Continual_Reinforcement_Learning.pdf`
is implemented and **working in the exact-gradient setting**. The core claim is
demonstrated: on a 2-task gridworld the constrained global policy retains both
tasks while an unconstrained baseline forgets the newest one.

- 19/19 tests pass (`pytest -q`, ~5 s CPU). Includes an exact-vs-finite-
  difference gradient check, Monte-Carlo/exact agreement, dual dynamics, and an
  end-to-end acceptance test comparing constrained vs unconstrained retention.
- `configs/gridworld_exact.yaml` is the canonical demo. Result:
  `eval_matrix = [[0.535, 0.106], [0.556, 0.555]]` (rows = after task 1 / after
  task 2; cols = value on task 1 / task 2). Unconstrained baseline collapses
  task 2 to ~0.05.
- Figures (PNG + SVG) via `python -m analysis.plots --run <dir>`: dual
  trajectories, gap sequences (cycling diagnostic), forgetting matrix, retention.

## Environment

Local dev used a venv built on the `rlclass` conda env (torch 2.11) with
`gymnasium`, `pyyaml`, `pytest`, `matplotlib` added:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Hyperion, `pip install -r requirements.txt` into a fresh env is enough
(CPU torch is fine for the gridworld/CartPole tiers; the exact tier is the
fastest way to sanity-check any change). No GPU needed until the Atari tier.

## What changed in the last session (derivation update)

The derivation was revised; the code was updated to match. If you compare
against an older mental model, note:

1. **Constraint is now a per-task, one-sided, *squared* hinge** (eqs 7, 11),
   not a linear aggregate. Local carries one multiplier `λ_i` per past task;
   global carries a single `μ`. Penalty applies only where the trained policy
   is *below* its frozen reference.
2. **Weights are uniform** `ω_i = 1/k`.
3. **Replay-free via env access.** Past-task values come from fresh rollouts in
   the old environments, not a stored buffer. This is now an explicit
   assumption in the derivation (Setup).
4. **`ε` is in squared-value units**: tolerated value gap `g` ↔ `ε = g²`. The
   dual step size must be large because the squared constraint value is tiny
   (gridworld uses `duals.lr: 8.0`).

## Open issues, most important first

### 1. Warm-start saturation (blocks 3+ task sequences)

With a shared, non-task-conditioned policy, `θ⁽⁰⁾ = φ` (eq 2) inherits an
increasingly saturated softmax as tasks accumulate. By the third task, policy
gradient escapes the inherited init too slowly and the new task barely learns
(`gridworld_three_task.yaml`, task 3 ≈ 0.01), even though task 3 reaches ~0.75
from a fresh init. This is saturation, not a bug, and it is the first thing to
fix before any multi-task result.

Candidate fixes, cheapest first:
- **Task-conditioned policy** (`policy.kind: mlp`, `task_conditioned: true`) —
  gives the new task its own head capacity; already implemented, needs testing.
- **Stronger / annealed entropy bonus** to keep the inherited policy escapable.
- **Periodic policy softening** (scale logits toward zero) at each task boundary.
- Reconsider whether the local *must* start exactly at `φ`; a partial or
  softened warm-start may keep eq-2's guarantee approximately while restoring
  plasticity. This touches the theory — flag to the author.

### 2. Rollout cost (settled in principle, tune in practice)

The derivation commits to fresh rollouts in old envs (no buffer, no OPE
needed). Cost scales with the number of past tasks `k`. In-repo mitigations:
- Frozen references are estimated **once per phase** with a large batch
  (`estimator.episodes_per_ref`) and held constant — this is the dominant saving.
- `trainer.past_task_sampling: sample` uses **one past task per step**, rescaled
  to stay unbiased (A-GEM-style single-constraint sampling), giving O(1) per-step
  cost instead of O(k).

If env re-instantiation ever becomes unavailable (true lifelong setting), the
literature route is a CLEAR-style per-task buffer + fitted-Q evaluation (Le et
al. 2019) or V-trace-corrected returns; a `frozen_surrogate` estimator stub is
in place (`crl/estimators/surrogate.py`) for that. **Do not build it without
asking** — it is a design decision with paper-level consequences, and the
current env-access assumption makes it unnecessary for the planned benchmarks.

### 3. Alternation stability

The pair can cycle (each phase moves the other's reference). Always inspect the
`gaps` figure. No convergence theory yet — this is the intended research
contribution. Mitigations to try if cycling appears: shorter phases, Polyak-
averaged references, the PID dual controller (`duals.kind: pid`).

## Recommended next experiment (lowest cost, highest signal)

Do **not** start at Atari. In order:

1. **Fix saturation on `gridworld_three_task`** (exact estimator). Turn on
   task-conditioning, confirm all three tasks are learned and retained, and that
   the constrained run beats the unconstrained baseline on average retained
   value. This validates the method beyond 2 tasks at essentially zero compute.
2. **Sweep `ε` and `duals.lr`** with `experiments/sweep.py` on the exact tier
   (`configs/sweeps/eps_grid.yaml`) across 3 seeds; produce the constraint-
   strength ablation (paper Fig 4). Exact estimator = seconds per run.
3. **`gridworld_sampled`** to confirm the story survives REINFORCE noise.
4. **`cartpole_family`** (MLP, sampled) — first non-tabular tier; expect to
   retune `duals.lr` and `ε` (watch `μ`/`λ` magnitudes on the first run).
5. Only then MiniGrid, then a 3-game Atari subset (Pong → Boxing → third),
   which needs an actor-critic estimator (new backend in `crl/estimators/`).

## How to add things (the repo is registry-driven)

- **New task family:** subclass `TaskFamily` in `crl/envs/`, register in
  `crl/envs/__init__.py::FAMILY_REGISTRY`. Tabular families also expose
  `(P, r, ρ)` tensors to unlock the exact estimator.
- **New estimator (e.g. actor-critic for Atari):** implement
  `ValueEstimator.evaluate` + `.surrogate_objective` in `crl/estimators/`,
  register in `ESTIMATOR_REGISTRY`. The trainer needs no changes.
- **New policy / dual controller:** same pattern under `crl/policies/` and
  `crl/duals/`.

Configs select components by name; no cross-component coupling. Keep all
hyperparameters in `configs/` (no magic numbers in scripts) and log the config
snapshot with every run (already automatic via `RunLogger`).

## Cluster notes (Hyperion)

- `experiments/sweep.py --index N` runs only the N-th grid point, so a sweep
  maps cleanly onto an sbatch array job. `--dry-run` prints the plan.
- Each run writes a self-describing directory under `results/<name>_seed<k>/`
  (`config.yaml`, `logs.jsonl`, `eval_matrix.json`, `final_policy.pt`,
  `figures/`). `results/` is gitignored.
- Reproducibility: `set_seed` covers torch/numpy/random/cuda; same seed → same
  eval matrix (enforced by `tests/test_config_and_seeding.py`).

## Docs to keep in sync

`docs/citation_corrections.md` lists 8 attribution fixes and one content-claim
error (Progress & Compress "transfer rises with tasks" is wrong — it is flat)
found by auditing `crl_literature_summary.pdf`. Apply before the paper cites
them.
