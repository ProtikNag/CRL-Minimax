# GridWorld (50-task, shared-head) — 6-method × 3-seed comparison

Fixed-capacity / no-growth regime. PERF/FWT/BWT normalized (0=random, 1=expert-ceiling); mean ± 95% CI over 3 seeds. FWT paired to seed-matched from-scratch baseline.

| Method | PERF ↑ | FWT ↑ | BWT ↑ | Forgetting ↓ | seeds |
|---|---|---|---|---|---|
| Ours (min-max) | 0.596 ± 0.017 | +0.077 | +0.358 ± 0.069 | 0.112 ± 0.019 | 3 |
| CKA-RL | 0.470 ± 0.099 | +0.162 | -0.187 ± 0.097 | 0.237 ± 0.069 | 3 |
| CReLUs | 0.328 ± 0.107 | +0.179 | -0.359 ± 0.140 | 0.391 ± 0.141 | 3 |
| Finetune | 0.261 ± 0.127 | +0.142 | -0.390 ± 0.131 | 0.430 ± 0.119 | 3 |
| CbpNet | 0.258 ± 0.074 | +0.156 | -0.430 ± 0.081 | 0.467 ± 0.081 | 3 |
| Baseline (scratch) | 0.158 ± 0.153 | — | -0.506 ± 0.174 | 0.559 ± 0.165 | 3 |
