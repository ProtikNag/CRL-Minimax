# Handoff

State of the project and what to do next. Written for a fresh session (on
Hyperion or elsewhere) that needs to continue without the prior context.

## Where things stand

The full method from `docs/Objective_for_Continual_Reinforcement_Learning.pdf`
is implemented and **working with a real neural network on a 3-task sequence**
(exact estimator). Core claim demonstrated: the constrained global policy
retains every task; the unconstrained baseline forgets the newest one.

- 21/21 tests pass (`pytest -q`, ~20 s CPU). Includes an exact-vs-finite-
  difference gradient check, Monte-Carlo/exact agreement, dual dynamics, and
  two end-to-end acceptance tests (tabular 2-task, and multi-head neural
  3-task).
- **Headline (neural) result**, `configs/gridworld_nn_three_task.yaml`, a
  multi-head MLP (shared trunk + one head per task) on 3 gridworld tasks:
  constrained final row ≈ `[0.83, 0.83, 0.83]` (all retained, 100% of expert);
  unconstrained (`duals.lr: 0`) ≈ `[0.83, 0.83, 0.20]` (newest task forgotten,
  24% retained). Reproduce both + all figures with one command:
  `python -m experiments.compare_constraint --config configs/gridworld_nn_three_task.yaml --name nn_three_task`
- The tabular 2-task `gridworld_exact` remains the smallest sanity demo.
- **Baselines** (all same network/tasks/estimator, only the procedure differs;
  `python -m experiments.baseline_comparison --config <cfg> --name <name>`):
  naive sequential fine-tuning of one network forgets the OLDEST task
  (49% / 100% / 100% success); the constraint-off ablation forgets the NEWEST
  (100% / 100% / 69%); joint multi-task is the upper bound (all 100%). Ours
  matches the upper bound. The two standard failures are in opposite directions,
  which is the core narrative. Baseline trainers live in `crl/baselines.py`.
- **Figures** land under `reports/<name>/` (tracked in git; raw runs stay in
  `results/`, gitignored). The committed bundle for the neural experiment is in
  `reports/nn_three_task/` (PNG + SVG, per-method diagnostics, retention CSV).

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

### 1. Warm-start saturation (was blocking 3+ tasks; addressed by task heads)

With a *shared*, non-task-conditioned policy, `θ⁽⁰⁾ = φ` (eq 2) inherits an
increasingly saturated softmax; by the third task the new task barely learns
(`gridworld_three_task.yaml`, tabular, task 3 ≈ 0.01) even though task 3 reaches
~0.75 from a fresh init. It is saturation, not a bug.

**Resolved for the neural setting** by the multi-head policy
(`policy.kind: multihead`): a shared Tanh trunk with one output head per task,
selected by the task id. Each task gets its own output mapping, so there is no
saturated shared logit to escape; the shared trunk still carries transfer and
is what the constraint protects. Result on the 3-task gridworld:
`[0.83, 0.83, 0.83]` (all learned and retained).

Two secondary notes for the write-up:
- **Larger local LR is a real but brittle partial fix** for the shared-policy
  case: raising `lr_local` from 0.5 to ~20 recovers tabular task 3 from 0.28 to
  0.83 (bigger steps escape the saturated init). It is fragile under sampling
  noise and does not fix the shared-representation root cause, so prefer heads.
  A warm-up LR boost per new task is a reasonable practical add if needed.
- **Shared vs disjoint state spaces.** The gridworld deliberately *shares* the
  state space across tasks (same cells, different goal), so the same state can
  demand different actions — a hard, adversarial conflict. Different Atari games
  have nearly *disjoint* observations, so that exact conflict is rare there;
  forgetting in Atari comes instead from shared *parameters* being overwritten.
  Task heads / conditioning help in both regimes; the mechanism differs. Keep a
  shared-state family (gridworld/MiniGrid) in the benchmark precisely because it
  is the harder stress test.

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

The neural 3-task result is done (exact estimator). Do **not** jump to Atari.
In order:

1. **`gridworld_nn_three_task_sampled`** — the single most valuable next run.
   Same multi-head network and tasks, but the Monte-Carlo estimator, so it is
   the *first test with real environment rollouts and sampling noise* and it
   exercises the past-task rollout machinery. The plumbing is verified; the
   hyperparameters are a starting point (retune `duals.lr` and
   `episodes_per_grad` from the `μ`/`λ` and gap figures). One command:
   `python -m experiments.compare_constraint --config configs/gridworld_nn_three_task_sampled.yaml --name nn_three_task_sampled`
2. **Sweep `ε` and `duals.lr`** with `experiments/sweep.py` on the exact neural
   tier (`configs/sweeps/eps_grid.yaml`) across 3 seeds; produce the
   constraint-strength ablation (paper Fig 4). Exact estimator = seconds per run.
3. **`cartpole_family`** (multi-head MLP, sampled) — first continuous-control
   tier; expect to retune `duals.lr` and `ε`.
4. Only then MiniGrid (another shared-state family), then a 3-game Atari subset
   (Pong → Boxing → third), which needs an actor-critic estimator (new backend
   in `crl/estimators/`).

## Generating figures

- Full bundle for an experiment (runs constrained + baseline, all figures +
  tables into `reports/<name>/`):
  `python -m experiments.compare_constraint --config <cfg> --name <name>`
- Single-run diagnostics only:
  `python -m analysis.plots --run results/<run_dir>`
- Conceptual figures only (design-space map, method schematic):
  `python -m analysis.schematics --out <figures_dir>`

Every figure is written in both PNG and SVG into split `png/` and `svg/`
subfolders, with titles, axis labels, legends, and the academic palette
(green = ours, red = unconstrained baseline). `reports/` is tracked in git so
figures can be committed on HPC and pulled locally.

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
