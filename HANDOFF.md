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

## ►► CURRENT STATE (2026-09-08)

### Results so far (seed 0)
- **Ours (min-max), retention vs the local reference:** Qbert 91%, Pong 99%,
  SpaceInvaders 68%, Breakout 39%, **Boxing −27%**. The retention gate rescues the
  oldest task (Qbert), but the final SpaceInvaders consolidation still wipes Boxing —
  a targeted **SI⟂Boxing interference**, not a global collapse.
- **CLEAR baseline** retains better and more evenly on this setup (from the V2
  comparison, seed 0): CLEAR held Qbert ~perfectly across all 5 tasks.

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

### Order-sensitivity run (IN PROGRESS)
`configs/atari5_v5_order2.yaml` (in the clone) = the same min-max method with the
**reversed** sequence SpaceInvaders→Boxing→Breakout→Pong→Qbert. **Job 21890786** on
`dgx_aic`. Tests whether the interference is order-driven (does a *different* game get
wiped when the order flips?). So far SpaceInvaders is fully retained through the
Boxing consolidation; needs tasks 4–5 to conclude.

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
1. **Finish the order-sensitivity run** (job 21890786) → compare its forgetting matrix
   to the canonical order (is Boxing's collapse order-specific?).
2. **Assemble the comparison** once the joint runs + order2 are in: joint ceiling vs
   consolidated (ours) vs CLEAR vs experts, per game; **recompute retention % against
   the joint ceiling** (not the experts). Gate figures through `visualization-expert`.
3. **Multi-seed** (≥3) — everything is seed 0; this is the single biggest gap.
4. (Deferred) longer task sequence; a retention metric that credits competent-but-
   sub-specialist play.

## ►► INFRA / REPO STATE
- **Branches:** `feature/updated-objective` = canonical (this branch, min-max).
  `feature/updated-objective-v2` = abandoned V2 critic-retention (preserved, don't
  resume). Old per-iteration Atari configs (v3/v4/v5, ppo_v3/4/5, atari4_*, CLEAR
  sweep B/C) were **removed** in cleanup; the canonical config is `atari5.yaml`.
- **Sibling clone `/work/pnag/CRL-Minimax-joint`** (a full clone of this branch) is
  where the **joint** and **order-sensitivity** runs execute, so they don't disturb
  this working tree. It has small local edits (`_train_joint` logs progress + saves
  `joint_iter{N}.pt`; configs `atari5_joint.yaml`, `atari5_joint_6m.yaml`,
  `atari5_v5_order2.yaml`; launcher `scripts/hpc_joint.sbatch`). **Retrieve the joint
  checkpoints/results from the clone, then it can be discarded.**
- **Stable GPU partitions:** `AI_Center_L40S` (node493), `dgx_aic` (dgx-1, 8×A100).
  Avoid V100 partitions (preempt/requeue). Atari throughput here is ~800 env-frames/s
  (env-bound), so a full 5-game run is many hours — budget accordingly.
- **Commits:** author `ProtikNag <protiknag08@gmail.com>`; end messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. `results/` and `experts/`
  are gitignored; `reports/`, `diagnostics/`, `pseudocode/` are tracked.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
