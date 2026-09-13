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

## ►► ACTIVE WORK (2026-09-12): CKA-RL (NeurIPS'25) comparison

Adding our min-max method as a new **row in the CKA-RL paper's Table 1
(PERF + FWT) and Table 3 (final-policy cross-task)**, on *their* benchmark
(`docs/papers/2025_Hu_CKA_RL_Continual_Knowledge_Adaptation`). Their "3 benchmarks"
are long single-game-variant sequences: **SpaceInvaders 10 modes, Freeway 8 modes,
Meta-World CW20 20 tasks** (38 tasks, ∆=1e6 steps/task; PPO for Atari, SAC for
Meta-World; "Baseline" = train-from-scratch-per-task, the FWT reference).

- **All this work lives in a SEPARATE clone `/work/pnag/CKA-RL-compare`, branch
  `ours-minmax-row`** — the main repo here is untouched by it. NOTE its git remote
  is the *upstream authors'* repo (`Fhujinwu/CKA-RL`, no push access); pushing the
  branch needs a personal fork (e.g. `ProtikNag/CKA-RL`).
- **Ports (both `code-verifier: PASS`):** Atari PPO `experiments/atari/run_ours.py`
  + `models/ours.py`; SAC/Meta-World `experiments/meta-world/run_sac_ours.py` +
  `models/ours.py` + `eval_final_policy_metaworld.py`. Single-process full-sequence
  runners mirroring the canonical method: task0 → per-task **local clone-from-global**
  specialist → **global consolidation** (one-sided squared value constraint, dual μ,
  two-timescale, retention-gated early stop). Speed variants: `--consolidate-mode
  needy` (targeted — replay only past tasks below their retention bar + current),
  `--vector-env async` (Atari only), `global_iters` cap; relative dual tolerance
  `eps_eff=(tol_frac·V_k^L)²`. Crash-safe observability (status.json / progress.jsonl
  / retention_history / checkpoints / `--resume`).
- **Running (seed 0):** SI-Ours + SI-Baseline (∆=1e6), Freeway-Ours + Freeway-Baseline
  (∆=1e6); Meta-Ours + Meta-Baseline at **reduced ∆=300k, global_iters=400**
  (single-env SAC only ~51 SPS → 1e6 would be ~a week; **300k is a DISCLOSURE item
  for the Meta row**). ETA: Freeway ~4-5h, SI ~10h, Meta ~2-3 days.
- **FIRST RESULT — Freeway (seed 0, PROVISIONAL), Freeway-Ours DONE:** Table-1
  **PERF 0.753** (CKA-RL 0.792, CompoNet 0.763, FT-N 0.753 — Ours mid-pack,
  *competitive, not better*), **FWT 0.661** (CKA-RL 0.743). Per-mode final-policy
  **retention ≈ 81%** of local specialists (3-ep greedy, noisy — clean 100-ep GPU
  eval pending), **no catastrophic forgetting** (worst mode 66%). So Freeway both
  learns competitively and retains well. **Not established as "better than the
  paper"**: on the one comparable metric (PERF) ours is slightly below CKA-RL, and
  retention isn't in the paper's success-units yet. Figure: `CKA-RL-compare/reports/
  freeway/` (visualization-expert checked). SI-Ours + Meta-Ours still running.
- **Disclosure flags for the "Ours" row** (from the CL-expert review): live past-task
  env access (baselines have none), >2× frames/task, Table-1 PERF = plasticity while
  retention shows in Table-3, single seed, reduced Meta ∆.
- **NEXT:** when a benchmark finishes → Atari: `gather_rt_results` +
  `process_results` + `eval_final_policy`; Meta: `extract_results` +
  `process_results` + `eval_final_policy_metaworld` → assemble the Table-1 + Table-3
  rows. Then multi-seed (≥3). **Full detail + job IDs in the auto-memory
  `cka-rl-comparison.md`.**

---

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
1. **Finish CLEAR reversed** (job 21906243, in the clone) → complete the order-sensitivity
   2×2 and add a CLEAR-reversed panel to `reports/order_sensitivity/` + the dashboard.
   Compare per-game to canonical CLEAR (Breakout→0) and to ours-reversed.
2. **Run `report/verify_dashboard.py`** on a machine with glibc ≥2.27 + Chromium (NOT
   possible on this cluster — see gotchas). The dashboard was static-verified only.
3. **Multi-seed** (≥3) — everything is seed 0; this is the single biggest gap, and
   order-sensitivity showed single-order/seed results can mislead.
4. (Deferred) longer task sequence; a retention metric that credits competent-but-
   sub-specialist play.

## ►► INFRA / REPO STATE
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
