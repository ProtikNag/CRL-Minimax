# atari4_v6_full — first full 1-seed min-max run (Part A, experts NOT stored)

Config `configs/atari4_v6_full.yaml`, branch `feature/updated-objective`, seed 0.
Order **Qbert → Boxing → Pong → Breakout** (Freeway replaced by Qbert). Budgets
task1/local = 3000, global = 2000; `constraint_episodes=16`, `past_task_sampling=all`.
Ran ~12.6 h on an L40S node (`aic_L40S_short`, non-preemptible). Checkpoints
`results/atari4_v6_full_seed0/global_after_task{k}.pt` + `final_policy.pt`.

## Forgetting matrix — global greedy-100 score, seen tasks (lower-triangular)
(threshold in parens; **bold** = just-learned diagonal)

| after ↓ | Qbert (4000) | Boxing (90) | Pong (18) | Breakout (50) |
|---|---|---|---|---|
| T1 Qbert    | **4217** |         |         |          |
| T2 Boxing   | 14330    | **17.5** |         |          |
| T3 Pong     | 5353     | −1.6    | **19.5** |          |
| T4 Breakout | 7148     | −3.6    | 19.8    | **53.7** |

## Read
- **Pong**: near-perfect retention (19.5 → 19.8; max 21).
- **Breakout**: learned (53.7 ✓).
- **Qbert**: learned + net-retained (4217 → 7148; noisy, spiked to 14330 mid-run).
- **Boxing**: the weak spot, on two counts:
  1. **Under-learned in the local phase** — ran the *full* 3000-iter cap yet was
     *still climbing* (greedy 7.8 → ~36–45, not plateaued) vs the 90 threshold.
     Boxing learns slowly here (θ⁰=φ init from a Qbert trunk drags it).
  2. **Then forgotten** — decayed to negative (−1.6, −3.6) under later consolidation.
     Compounded under-learning → collapse.

**Takeaway:** the min-max held retention where the local model actually mastered
the task (Pong; Qbert-ish); the one game the local phase failed to learn (Boxing)
is exactly the one that collapsed. Next rerun: larger local budget for slow games
(or fresh-init local). Full data: `results/atari4_v6_full_seed0/{eval_matrix.json,logs.jsonl}`.

## Wall-clock breakdown (from logs.jsonl t_wall)
task1 Qbert ~37 min (early-stop) · T2 Boxing local ~2.5 h (full cap) + global ~3.8 h
· T3 Pong local ~25 min (early-stop) + global ~1.5 h · T4 Breakout local ~53 min
(early-stop) + global ~2.85 h. Total ~12.6 h. Live per-iter on L40S: local ~2.3 s,
global ~4.6–9.2 s (grows with #past tasks under `past_task_sampling=all`).
