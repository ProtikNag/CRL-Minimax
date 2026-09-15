# GridWorld (shared-head, 50-task) — results (seeds 0-1 complete; seed 2 in progress)

Fixed-capacity/no-growth regime: single shared actor+critic head (mlp_ac, task-conditioned),
no per-task heads. CKA-RL = faithful reimpl (shared trunk + bounded pool + per-task alpha, no growth);
CompoNet excluded (it grows the net). FWT vs from-scratch baseline (per seed).

## Aggregated over COMPLETE seeds (ours 3; finetune/baseline/CKA-RL 2 = seeds 0,1)

| Method | PERF | Forgetting | BWT | FWT | seeds |
|---|--:|--:|--:|--:|:--|
| **ours** | 0.596 ± 0.006 | 0.112 ± 0.006 | +0.358 ± 0.023 | +0.080 ± 0.052 | 3 |
| CKA-RL | 0.490 ± 0.019 | 0.226 ± 0.021 | -0.168 ± 0.019 | +0.181 ± 0.015 | 2 |
| finetune | 0.261 ± 0.042 | 0.430 ± 0.039 | -0.390 ± 0.043 | +0.147 ± 0.018 | 3 |
| baseline | 0.193 ± 0.009 | 0.520 ± 0.004 | -0.466 ± 0.000 | 0 (ref) | 2 |

## Per-seed numbers in metrics.json; raw + tidy CSVs (forgetting_matrix, learning_curves, duals, phases) per run dir.
Seed 2 of finetune/baseline/CKA-RL still running -> full 3-seed CIs land shortly (ours already 3/3).

## Read: retention ours >> CKA-RL > finetune > baseline; ours only method with positive BWT. FWT CKA-RL~finetune>ours (plasticity-retention tradeoff).
## Disclosures: footprint ours=fixed shared head, CKA-RL=fixed trunk+bounded pool+per-task alpha (no growth). Ours re-simulates past envs; CKA-RL/finetune train only current task.