# GridWorld (shared-head, 50-task) — PARTIAL results

**PARTIAL / IN-PROGRESS snapshot** — some runs not yet at 50/50; 'final' = latest
completed row, so forgetting/BWT are lower bounds and will edge up as the last tasks
land. FWT is vs the from-scratch baseline. Numbers will be refreshed on completion.

Setting: shared actor+critic head (mlp_ac, task-conditioned) — NO per-task heads, so
forgetting is forced (the fixed-capacity / no-growth regime). CompoNet dropped;
CKA-RL run as our faithful reimpl (shared trunk + bounded pool + per-task alpha).

| Method | tasks | done | PERF | Forgetting | BWT | FWT |
|---|--:|:--:|--:|--:|--:|--:|
| ours seed0 | 45 | · | 0.618 | +0.096 | +0.375 | +0.123 |
| ours seed1 | 44 | · | 0.615 | +0.093 | +0.350 | +0.055 |
| ours seed2 | 50 | Y | 0.596 | +0.116 | +0.383 | +0.079 |
| finetune | 50 | Y | 0.309 | +0.386 | -0.344 | +0.163 |
| baseline | 50 | Y | 0.184 | +0.525 | -0.465 | 0 (ref) |
| CKA-RL | 41 | · | 0.385 | +0.304 | -0.261 | +0.178 |

## Per-run granular data (for detailed viz)
Each run dir has raw run.json/progress.jsonl/eval_matrix.json + tidy CSVs:
- forgetting_matrix.csv (every end-of-task row x every seen task; raw+normalized)
- learning_curves.csv (within-phase FWT: per-task greedy score vs iter/frames)
- duals.csv (mu/shortfall/coeff — ours only) · phases.csv (per-phase iters/frames/wall)

## Read (preliminary)
- Retention: ours >> CKA-RL > finetune > baseline. Ours ~0.60 PERF, ~0.10 forgetting,
  strong +BWT (~0.36). Finetune collapses (per-task-head crutch removed).
- CKA-RL partially retains via its pooled actor head, but the shared trunk it cannot
  grow drifts -> ~2x more forgetting than ours (the no-growth story).
- FWT: CKA-RL ~ finetune > ours (plasticity-vs-retention tradeoff).
- Contrast: per-task-head reports/gridworld/ had finetune COMPETITIVE (PERF 0.644);
  removing separate heads is what exposes forgetting.

## Disclosures (for the paper table)
- Footprint: ours = fixed shared head; CKA-RL = fixed trunk + bounded pool(5) +
  tiny per-task alpha; no network growth (CompoNet, which grows, is excluded).
- Ours re-simulates past environments during consolidation (live past-task access);
  CKA-RL/finetune train only on the current task. Disclose in captions.