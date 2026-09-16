# GridWorld (shared-head, 50-task) — 6-method comparison

Fixed-capacity / no-growth regime (single shared actor+critic head, task-conditioned).
3 recent methods (CKA-RL, CbpNet, CReLUs) + 2 basic (finetune, baseline) vs ours.
FWT vs from-scratch baseline (per seed). Aggregated over COMPLETE seeds.

| Method | PERF | Forgetting | BWT | FWT | seeds |
|---|--:|--:|--:|--:|:--|
| **ours** | 0.596 ± 0.006 | 0.112 ± 0.006 | +0.358 ± 0.023 | +0.077 ± 0.051 | 3 |
| CKA-RL | 0.470 ± 0.032 | 0.237 ± 0.023 | -0.187 ± 0.032 | +0.162 ± 0.031 | 3 |
| CbpNet | 0.284 ± 0.000 | 0.431 ± 0.000 | -0.394 ± 0.000 | +0.175 ± 0.000 | 1 |
| CReLUs | 0.329 ± 0.000 | 0.407 ± 0.000 | -0.371 ± 0.000 | +0.251 ± 0.000 | 1 |
| finetune | 0.261 ± 0.042 | 0.430 ± 0.039 | -0.390 ± 0.043 | +0.142 ± 0.017 | 3 |
| baseline | 0.158 ± 0.050 | 0.559 ± 0.054 | -0.506 ± 0.057 | 0 (ref) | 3 |

(ours/CKA-RL/finetune/baseline = 3 seeds; CbpNet/CReLUs = 1 seed so far, s1/s2 running -> 3-seed CIs soon.)

## Read: ours >> CKA-RL (retention) > CbpNet/CReLUs (plasticity methods, forget ~like finetune, high FWT) ~ finetune > baseline. Ours only method with positive BWT.
## Footprint (all no-growth): ours=fixed head; CKA-RL=fixed trunk+bounded pool+alpha; CbpNet=fixed+unit resets; CReLUs=fixed+CReLU act. Disclosure: ours re-simulates past envs.
## Per-run raw + tidy CSVs (forgetting_matrix, learning_curves, duals, phases) per dir.