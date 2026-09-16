# GridWorld (shared-head, 50-task) — 6-method comparison

Fixed-capacity / no-growth regime: single shared actor+critic head (mlp_ac,
task-conditioned), NO per-task heads. All 6 methods fixed-capacity: ours (min-max
constraint), CKA-RL (bounded pool + per-task alpha), CbpNet (unit resets), CReLUs
(CReLU activation), finetune, baseline. FWT vs from-scratch baseline (per seed).

## Aggregated over COMPLETE seeds

| Method | PERF | Forgetting | BWT | FWT | seeds |
|---|--:|--:|--:|--:|:--|
| **ours** | 0.596 ± 0.006 | 0.112 ± 0.006 | +0.358 ± 0.023 | +0.077 ± 0.051 | 3 |
| CKA-RL | 0.470 ± 0.032 | 0.237 ± 0.023 | -0.187 ± 0.032 | +0.162 ± 0.031 | 3 |
| CbpNet | (running) | | | | 0 |
| CReLUs | (running) | | | | 0 |
| finetune | 0.261 ± 0.042 | 0.430 ± 0.039 | -0.390 ± 0.043 | +0.142 ± 0.017 | 3 |
| baseline | 0.158 ± 0.050 | 0.559 ± 0.054 | -0.506 ± 0.057 | 0 (ref) | 3 |

(ours/CKA-RL/finetune/baseline = 3 seeds complete; CbpNet/CReLUs still running -> their rows fill in as seeds finish.)

## Read: retention ours >> CKA-RL > (CbpNet/CReLUs pending) > finetune > baseline; ours only method with positive BWT. FWT CKA-RL~finetune>ours (plasticity-retention tradeoff).
## Footprint (all no-growth): ours=fixed head; CKA-RL=fixed trunk+bounded pool+alpha; CbpNet=fixed+unit resets; CReLUs=fixed+CReLU. Disclosure: ours re-simulates past envs; others train only current task.
## Per-run raw + tidy CSVs (forgetting_matrix, learning_curves, duals, phases) per dir.