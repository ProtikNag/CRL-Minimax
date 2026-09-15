# GridWorld (shared-head, 50-task) — results

Fixed-capacity / no-growth regime: ONE shared actor+critic head (mlp_ac, task-
conditioned), NO per-task heads -> forgetting is forced. CompoNet dropped (it grows
the network); CKA-RL = our faithful reimpl (shared trunk + bounded pool + per-task
alpha, no growth). Reported eval greedy; FWT vs the from-scratch baseline (per seed).

## Complete runs (ours = 3 seeds mean±std; finetune/baseline/CKA-RL = seed 0 done; their seeds 1-2 still running)

| Method | PERF | Forgetting | BWT | FWT |
|---|--:|--:|--:|--:|
| **ours (3 seeds)** | 0.596 ± 0.006 | 0.112 ± 0.006 | +0.358 ± 0.023 | +0.045 ± 0.077 (n=3) |
| CKA-RL (s0) | 0.509 | +0.205 | -0.149 | +0.197 |
| finetune (s0) | 0.309 | +0.386 | -0.344 | +0.163 |
| baseline (s0) | 0.184 | +0.525 | -0.465 | 0 (ref) |

## Footprint / disclosures (for the paper)
- ours = FIXED shared head; CKA-RL = fixed trunk + bounded pool(5) + tiny per-task alpha; both no network growth (CompoNet, which grows, excluded).
- ours re-simulates past environments during consolidation (live past-task access); CKA-RL/finetune train only on the current task — disclose in captions.
- CKA-RL's forgetting here is honest for the no-growth regime: its pooled head retains actor knowledge but the shared trunk it can't grow drifts.

## Read
- Retention: ours (PERF ~0.60, forget ~0.11, BWT +0.36) >> CKA-RL (0.51/0.21) > finetune (0.31/0.39) > baseline (0.18/0.53).
- Ours is the ONLY method with positive BWT (consolidation improves old tasks).
- FWT: CKA-RL ~ finetune > ours (plasticity-vs-retention tradeoff).
- Contrast with the per-task-head run (reports/gridworld/): there finetune was COMPETITIVE (PERF 0.644) because separate heads relieve forgetting; the shared head exposes it.

## Granular data (per run dir): raw run.json/progress.jsonl/eval_matrix.json + tidy CSVs
(forgetting_matrix, learning_curves, duals[ours], phases). 3-seed CIs for all methods land when seeds 1-2 finish (STEP 2).