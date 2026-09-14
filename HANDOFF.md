# Handoff

Current state of the project and what to do next, for a fresh session. Read
`README.md` first (problem, method, math); this file is status + next steps only.

---

## The method (canonical, min-max consolidation)

Sequential continual RL over Atari games with a shared-trunk multi-head net. For
each new task: a **local** phase (standard PPO on the current game, from θ⁰=φ)
produces a frozen per-task **reference** (`local_after_task{k}.pt`); a **global**
phase consolidates — maximizing past-task return while a one-sided squared value
constraint (dual **μ**) keeps the current game at its local level. Stabilizers:
**μ-cap** (`duals.max_value`) and a **retention-gated global early-stop** (stop only
when EVERY seen task ≥ `global_retention_frac` of its local reference, else run to
the cap). REINFORCE backend = the exact-gradient theory harness; **PPO backend =
Atari** (`trainer.kind: ppo`, `ppo.method: constrained`).

**Binding rules:** reported eval is **greedy, 100 rollouts** (never stochastic);
Atari envs are **uncapped** (`max_steps: 0`). Per-task **reference = the LOCAL
specialist** (greedy-100 `local_greedy`), not stored experts.

**Configs:** `configs/atari5.yaml` (ours) · `configs/atari5_clear.yaml` (CLEAR
baseline, apples-to-apples). Launch: `sbatch scripts/hpc_atari_worker.sbatch
<config> <seed>`. (There is no more V1–V5 version numbering — this *is* the method;
older per-iteration configs were removed in the 2026-09-08 cleanup.)

---

## ►► READ THIS FIRST — ICLR sprint (opened 2026-09-14, deadline ~2026-09-25)

Everything below the sprint section is background. This section is the plan.

**Where the paper stands.** The method works and the line-1 and line-2 figures are
built and committed under `reports/final/`. What is missing is breadth: one seed,
two baselines, a stale GridWorld tier, and one config bug that holds our own
numbers down. The sprint closes those in priority order.

### Ground rules for every run launched from here

These are not suggestions. A run that violates them has to be redone, and there
is no time to redo runs.

1. **Log everything, to one schema.** `docs/LOGGING_CONTRACT.md` defines it.
   Ours, CompoNet, CKA-RL, CLEAR and the from-scratch baselines all emit the same
   `progress.jsonl`, so one metrics module computes PERF / FWT / BWT / forgetting
   / retention / compute for all of them. A method that cannot fill a field writes
   `null`; it does not invent its own key.
2. **Progress must be checkable mid-flight.** `status.json` heartbeat at least
   every 60 s, plus append-only `progress.jsonl`. Never wait until a run ends to
   discover it went wrong at task 2.
3. **Checkpoint after every task.** A failure on task 4 resumes from task 3,
   never from task 0. Resume restores optimiser **and dual state**, not just
   weights, or the method silently changes at the resume point.

### Priorities, as lanes (Hyperion has 2-3 GPUs; run these in parallel)

| Lane | Work | Status | Notes |
|------|------|--------|-------|
| **0 — now, eval only** | FWT baseline-parity check | **TODO** | Hours, no GPU. Highest value per hour on the list. See below. |
| **A** | GridWorld: new hard 50-task env, ours, 3 seeds | **TODO** | Should land same-day. Cheap tier, and the only one that gets error bars. |
| **B** | Atari: rerun ours, task-1 local budget fixed | **TODO** | Launch first, ~25-30 h. The paper's headline. |
| **C** | CompoNet: implement once → GridWorld check → Atari | **TODO** | Implement against GridWorld first; it is the cheap correctness check before 20 GPU-hours. |
| **C then** | CKA-RL: implement once → GridWorld check → Atari | **TODO** | Same pattern. |
| **spare** | Atari ours, seed 1 | opportunistic | Same config, unattended. Free insurance on the single-seed objection. Kill it if a GPU is needed. |
| **background** | Meta-World CW20 | RUNNING | job 21910655, reduced Δ=300k. Finishes on its own. |

**Deliberately deferred:** multi-seed Atari beyond the opportunistic seed 1.
The decision is to answer a seed complaint in rebuttal rather than spend the
sprint on it. GridWorld ×3 seeds carries the error-bar story in the meantime.

### Lane 0 — FWT baseline-parity check (do this first)

CKA-RL's forward transfer is `FT_i = (AUC_i − AUC_i^b) / (1 − AUC_i^b)`, and in
their Table 1 **every method shares one reference**: the Baseline row, whose own
FWT is 0.0000 by construction. We ran **our own** SI-Baseline and
Freeway-Baseline.

If our Baseline is *stronger* than theirs, our FWT is depressed for a reason that
has nothing to do with our method, and that alone could explain 0.650 against
CKA-RL's 0.775.

1. Compare our Baseline's PERF to the paper's: **SI 0.6314, Freeway 0.1247**.
2. If they differ materially, recompute **only the Min-Max row** against the
   paper's published Baseline AUC. Every other row is their published number and
   stays untouched.
3. The caption must then say ours' FWT is computed against *their* Baseline
   rather than our own run. That is legitimate; leaving it unsaid is not.

If they match, their numbers stand and no reimplementation is justified on that
ground alone.

### Lane A — the harder GridWorld

The existing GridWorld results were produced under the **previous objective** and
are not faithful to the current method. They are being replaced, not extended.

- **Family:** extend `biggrid` (already 50×50, `configs/biggrid_20task.yaml`).
- **50 tasks**, neural policy, **not** the tabular/exact estimator.
- **Difficulty comes from dynamics, not from grid area.** Growing the grid mostly
  buys longer episodes, which costs wall-clock without adding task diversity.
  Vary instead: obstacle density (always leaving a path), obstacle type,
  slipperiness / transition noise, and reward and penalty magnitudes.
- **Per-task heads**, matching Atari. But keep the **trunk deliberately narrow**
  relative to 50 heads — `gridworld_20task.yaml` already does this
  (`hidden_sizes: [64]`, *"bottleneck trunk → forces cross-task sharing"*). With
  a wide trunk and 50 heads the shared trunk stops being contested, forgetting
  has nowhere to live, and the benchmark comes back a null result.
- **`past_task_sampling: sample` is mandatory.** Global consolidation replays past
  tasks each iteration, so it is O(k); at task 50 that is ~50× task 1's
  per-iteration cost and the sequence is quadratic. The sampled estimator is
  unbiased and O(1) (`crl/trainer.py:136`).
- **3 seeds.** Cheap here, and it means the theory tier reports CIs even while
  Atari is single-seed.
- Needs a paired **from-scratch baseline run** for FWT (see the contract).

### Lane B — the Atari rerun, and the one bug behind it

`crl/ppo_continual.py:169` uses `task1_iters` for task 1, while
`local_iters_per_task` is only consulted at line 211, for tasks 2 onward. In
`configs/atari5.yaml` that is **1500 for task 1 against a `SpaceInvaders: 3000`
entry that never fires when SI is task 1**.

In the reversed order SI *is* task 1, so it trained on half the budget its own
config entry asks for. That is why the SI local specialist sits at 588 against a
Joint ceiling of 906 — a config bug, not a weakness in the method. The local
reference is the bar the global's μ-constraint must clear on the current task, so
a weak local caps the global.

**The fix:** train each specialist **to convergence (plateau on greedy-100),
with a hard cap**, rather than to a fixed iteration count.

Do **not** make "train until ≥ the joint ceiling" the rule, for three reasons:

1. It may not terminate. Joint trains on all five games with transfer between
   them; a single-task learner may never reach that score at any budget.
2. **It breaks the continual premise.** The joint model needs all five games at
   once. Making it a training target means the method consumes something a
   continual learner cannot have, and a reviewer will catch it.
3. It makes the ceiling circular. We normalise retention against Joint. If locals
   are trained *to* Joint by construction, Joint stops being an independent
   reference.

Convergence-to-plateau reaches the same place without referencing Joint at all,
and Joint stays an honest post-hoc check: *"every specialist meets or exceeds the
budget-matched multi-task ceiling."* That is a stronger sentence than the version
that trains to it.

**Scope of the rerun: same five games, same reversed order, ONLY the task-1 local
budget and the convergence rule change.** Changing the sequence at the same time
would make the difference unattributable. Plus the one logging change below.

### `eval_every` must be on — this is the FWT fix

`configs/atari5.yaml` sets `eval_every: 0`. Forward transfer is the **area under
task i's learning curve during its own training phase**, so without periodic
within-phase eval there is no `p_i(t)` and FWT cannot be recovered from a
finished run. Turn it on before launching lane B; there is no second shot at a
25-hour run.

**Correction to an earlier claim in this repo.** `reports/final/atari_reversed/`
previously said forward transfer was "not measurable in this study". That was
overstated. It is true only of the *zero-shot* Lopez-Paz form `R[i−1, i]`, which
per-task heads do make meaningless (a task's head is at random init until its
task arrives). The **AUC form that CKA-RL and Continual World actually use is
measurable**, needs no upper triangle, and is unaffected by per-task heads —
both the continual learner and the baseline start task i's head from scratch, and
what differs is the trunk, which is exactly what forward transfer measures.

**Before training new baselines:** `experiments/train_expert.py` already trains
from-scratch single-task experts from a shared init. Those *are* the FWT baseline.
Check whether their learning curves were logged — Atari's `AUC^b` may already be
on disk.

### Lane C — CompoNet and CKA-RL

Both are needed on the Atari sequence. The scientific point they serve: the
CKA-RL benchmark's tasks are **variants of one game**, which is a far easier
continual problem than **five distinct games** sharing a trunk. Showing how these
methods do on the harder setting is the argument.

- **Implement each once, run on both tiers.** Build against the GridWorld
  interface first — it is minutes per run, so it is the cheap correctness check
  before committing 20 GPU-hours on Atari.
- Papers go in `docs/papers/`. CKA-RL is already there. CompoNet needs fetching.
- A sloppy implementation is worse than none: we are adding rows to *their* table
  and the implementations will be checked.


## ►► CURRENT STATE (2026-09-11)

### Results so far (seed 0, canonical order Qbert→Pong→Breakout→Boxing→SpaceInvaders)
Per-game final greedy-100 (Qbert,Pong,Breakout,Boxing,SI):
- **Ours (min-max), `results/atari5_v5_seed0`:** [4075, 19.8, 51.8, **−25.8**, 765.8].
  Retention vs local: 91/99/39/**−27**/68% (mean 54%). Retention gate rescues Qbert;
  the final SI consolidation **wipes Boxing** (SI⟂Boxing interference), not a global collapse.
- **CLEAR (plain-net), `results/atari5_v5_clearA_equal_seed0`:** [4350, 21, **0.0**, 100, 800].
  CLEAR retains Qbert/Pong/Boxing/SI well but **forgets Breakout entirely (0)**.
  → **Ours and CLEAR each catastrophically forget a DIFFERENT game** (ours→Boxing, CLEAR→Breakout).
- Local refs (ours' specialists): [4467.8, 20.0, 132.7, 94.0, 1132.2].

### Joint ceiling (NEW — the fair upper bound)
Comparing the consolidated model to single-task **experts** is unfair (experts
balance one game). The honest ceiling is a **budget-matched jointly-trained model**
(all 5 games mixed, equal weight, no constraint) — "how well can ONE shared net do
on all games at once." Runs live in the sibling clone (see Infra):
- **6M-frames/game (30M total): DONE.** Final greedy-100
  (Qbert/Pong/Breakout/Boxing/SI) = **[4261, 20.7, 285, 67.5, 906]** — clears **all
  5 thresholds**. → a single net *can* play all five.
- **V5-frame-matched (59.8M total): stopped at it4500 (~46M frames), close enough to
  the target.** Last probe = [4430, 21.0, 218, 70.1, 1375]. Checkpoints every 500
  iters (`joint_iter{N}.pt`) + per-game probes (with cumulative frames/episodes) in
  the clone's `results/atari5_joint_seed0/`.
- The gap between this ceiling and the consolidated models = the true **cost of
  sequential learning / forgetting** — the right thing to normalize retention against.

### Order sensitivity (a headline finding) — 2×2 of {ours, CLEAR} × {canonical, reversed}
Reversed order = SpaceInvaders→Boxing→Breakout→Pong→Qbert. Same games/net/budget.
- **Ours, reversed (`results/atari5_v5_order2_seed0`, DONE, job 21890786):** by game
  [Qbert 4270, Pong 21, Breakout 200, Boxing 55.4, SI 712]. **Boxing's canonical
  catastrophe (−27% of local) becomes +56% when reversed** — no collapse. Retention vs
  the FIXED joint ceiling: **mean 51% (canonical) → 86% (reversed)**. So ours' forgetting
  is STRONGLY order-dependent; the Boxing loss is order-specific (Boxing learned 4th then
  wiped by SI; learned 2nd it survives).
- **CLEAR, reversed (`results/atari5_clear_order2_seed0`, RUNNING, job 21906243 on dgx-1):**
  at task 3/5 as of 2026-09-11. So far: after T1 SI 588, after T2 SI 533 + Boxing 100 (SI
  retained through Boxing). Watch whether Breakout (canonical CLEAR wiped it) is wiped again
  by a later task, i.e. whether CLEAR is more order-robust than ours. **Resume/monitor:
  `results/atari5_clear_order2_seed0/logs.jsonl` in the CLONE.**

### Paper figure set — `reports/final/` (DONE, pushed)

The only figures that are paper deliverables. Exploratory sets stay where they
were built (`reports/order_sensitivity/`, `reports/v5_clear_joint/`,
`diagnostics/`) and are not deliverables. Style spec for reproducing the look in
another repo: `reports/final/FIGURE_STYLE.md`, mirrored into the
`algorithmic-art-academic` skill section F.5b.

- `reports/final/atari_reversed/` — line 1, five-game reversed sequence:
  `forgetting_matrices`, `final_scores`, `backward_transfer_matrix`,
  `transfer_table`, `compute_cost`. All PNG (300 dpi) + SVG, all vector.
- `reports/final/cka_rl/` — line 2, CKA-RL benchmark: `table1_perf_fwt`,
  `final_policy_space_invaders`, `final_policy_freeway`. Per-mode source records
  in `raw/`, pushed from the cluster.

**Finding worth keeping in view:** the backward-transfer matrix shows ours'
forgetting is **non-monotone** — Boxing collapses to −1.12 under the Breakout
consolidation and recovers to −0.34, Space Invaders ends above where it was
learned, 3 of 10 cells positive — while CLEAR has **no positive cell anywhere**
and its two worst columns decline monotonically. Reads as consolidation actively
repairing a past task where replay only slows the bleed. Ours' single worst cell
is still worse than anything CLEAR does; the difference is it does not stay
there. Single seed.

### Visualization / dashboard (DONE, pushed)
- **Retention figures** `reports/v5_clear_joint/` and `reports/order_sensitivity/`
  (visualization-expert: FAITHFUL) — ours vs CLEAR vs joint; retention vs local + vs joint;
  order-sensitivity canonical vs reversed.
- **HTML results dashboard** `report/index.html` (commit `799be4c`): 10 figures / 6 groups
  (method_comparison, retention_forgetting, clear_buffer_sweep, order_sensitivity,
  training_dynamics, compute_cost), built with `report/acviz.py` (Track F skill) from the
  real runs, PNG+SVG each. Metrics grounded in `analysis/continual_metrics.py`.

### The "critic-based retention" variant (V2) — TRIED AND ABANDONED
A newer formulation (per-task λ_i critic constraints at anchor states, stored critics)
was implemented, verified (`code-verifier: PASS`), and run — it **forgot worse than
CLEAR and used ~3.7× the frames**. Diagnosed root causes: (1) the current-task hinge
coeff `2μ·shortfall` is unbounded and **dominates the shared-trunk gradient** (90–98%),
overwriting old-task features toward the new specialist; (2) the retention gradient is
**ineffective even at λ=cap** (the short-horizon, optimistically-bootstrapped value
residual doesn't restore greedy game competence); (3) an **anchor loophole** (anchors
from the degrading global trivially satisfy the bar once forgetting starts). It is
**preserved (committed) on branch `feature/updated-objective-v2`** for the record but
is NOT the main line. **Do not resume it without an explicit decision.**

---

## ►► NEXT STEPS

**The sprint section at the top of this file is the plan.** What remains here is
the residue that is not part of it.

1. **Run `report/verify_dashboard.py`** on a machine with glibc ≥2.27 + Chromium
   (not possible on this cluster — see gotchas). The dashboard was
   static-verified only. Low priority: `reports/final/` is the paper deliverable
   and does not depend on the dashboard.
2. **Fix the stale caption in `report/manifest.json`.** Its `compute_cost_wall`
   entry reads *"V5 is substantially slower than CLEAR"*, which contradicts its
   own numbers (15.1 h and 18.1 h against 36.4-37.3 h). Correct or drop it before
   the dashboard is shown to anyone.
3. (Deferred, post-deadline) multi-seed Atari beyond the opportunistic seed 1;
   a retention metric that credits competent-but-sub-specialist play.

### Done, do not redo

- CLEAR reversed finished; the order-sensitivity 2×2 is complete.
- `reports/final/` holds the paper figure set for both lines (see below).

## ►► INFRA / REPO STATE

- **Build the code graph first, on whatever machine you are on.**
  `graphify-out/` is **gitignored** — it is derived, AST-only and costs nothing,
  so it does not travel with the repo. Run `graphify update .` once before
  exploring; then `graphify query "<question>"` beats grepping for anything
  architectural. Currently 1600 nodes / 3363 edges locally.
- **Branches:** `feature/updated-objective` = canonical (this branch, min-max).
  `feature/updated-objective-v2` = abandoned V2 critic-retention (preserved, don't
  resume). Old per-iteration Atari configs (v3/v4/v5, ppo_v3/4/5, atari4_*, CLEAR
  sweep B/C) were **removed** in cleanup; the canonical config is `atari5.yaml`.
- **Sibling clone `/work/pnag/CRL-Minimax-joint`** (a full clone of this branch) is
  where the **joint**, **order-sensitivity**, and **CLEAR-reversed** runs execute, so they
  don't disturb this working tree. Local edits: `_train_joint` logs progress + saves
  `joint_iter{N}.pt`; configs `atari5_joint.yaml`, `atari5_joint_6m.yaml`,
  `atari5_v5_order2.yaml`, `atari5_clear_order2.yaml`; launcher `scripts/hpc_joint.sbatch`.
  Their JSON/log results were also copied into the main repo's `results/` (gitignored) for
  the dashboard. **Retrieve results from the clone, then it can be discarded.**
- **Stable GPU partitions:** `AI_Center_L40S` (node493), `dgx_aic` (dgx-1, 8×A100).
  Avoid V100 partitions (preempt/requeue). Atari throughput here is ~800 env-frames/s
  (env-bound), so a full 5-game run is many hours — budget accordingly.
- **ENVIRONMENT GOTCHAS (cluster is CentOS 7 / glibc 2.17 on every node):**
  - **Figures:** the viz pipeline (`report/acviz.py`) needs `plotly==5.24.1 + kaleido==0.2.1`
    here — kaleido 1.x / plotly 7 drive a modern Chrome that CANNOT run on glibc 2.17.
    (`requirements-viz.txt` allows these pins.) Install `--user`.
  - **Dashboard verification:** `report/verify_dashboard.py` uses Playwright/Chromium →
    **impossible anywhere on this cluster** (glibc 2.17, no container runtime). Static-verify
    (manifest valid, assets non-empty) and run the Playwright check off-cluster.
  - **Full `/tmp`:** some compute nodes (e.g. node335) have a tiny 2 GB `/tmp` that fills from
    other jobs, which breaks Claude Code's command-output capture (ENOSPC). Fix: start the
    session with `export CLAUDE_CODE_TMPDIR=/work/pnag/.claude-tmp` (dir on `/work`, which has
    space), or land on a node with free `/tmp`. Does NOT affect runs (they're on other nodes → `/work`).
- **Commits:** author `ProtikNag <protiknag08@gmail.com>`; end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. `results/` and `experts/`
  are gitignored; `reports/`, `diagnostics/`, `pseudocode/` are tracked.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
