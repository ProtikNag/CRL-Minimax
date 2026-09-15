# GridWorld (hard 50-task BigGrid) — results

Continual RL on 50 goal+dynamics tasks (per-task slip {0,.1,.2,.3}, obstacle density {0,.08,.15}), PPO actor-critic backend, shared narrow [64] trunk + per-task actor+critic heads. Task-incremental (task id at train+test). Metrics per docs/LOGGING_CONTRACT.md §6 from progress.jsonl / eval_matrix.json.

## Headline (3 seeds for ours; 1 seed finetune/baseline)

| Method | PERF | Forgetting | BWT | FWT |
|---|--:|--:|--:|--:|
| **Min-Max (ours)** | 0.545 ± 0.041 | 0.136 ± 0.029 | +0.507 ± 0.014 | 0.069 ± 0.035 |
| Naive fine-tune | 0.644 | 0.115 | -0.063 | 0.000 |
| From-scratch baseline | 0.055 | 0.636 | -0.631 | (ref) |

## Honest reading (preliminary)

- **Naive fine-tune is competitive-to-better on PERF** (finetune 0.644 vs ours 0.545) with comparable forgetting. On this task-incremental benchmark the **per-task heads relieve catastrophic forgetting** (each task keeps its own head; only the shared [64] trunk is contested), so the classic "ours prevents forgetting where naive collapses" contrast is **weak** here. This matches the continual-learning-expert's flag.
- **Ours' differentiator is strongly positive BWT (+0.51 vs −0.06)**: consolidation keeps improving already-seen tasks, whereas fine-tune drifts slightly negative. Old-task performance rises after later tasks — consolidation actively repairs, not just protects.
- **From-scratch baseline collapses (PERF 0.055)**: it trains each task on a fresh policy, so the final model scores ~random on earlier tasks. It is the FWT reference only, not a retention competitor.
- **FWT is small-positive** (ours ~0.07): mild forward transfer from the warm-started local specialist over from-scratch.

## Caveats

- Reported eval uses **30 greedy episodes** per task/cell, not the binding 100 (docs/LOGGING_CONTRACT.md). Numbers are usable but noisier than the rule; a 100-episode post-hoc re-eval from the saved per-task checkpoints is the exact fix (not run here per instruction).
- **finetune/baseline are single-seed**; only ours has 3-seed CIs.
- Retention early-stop is **local-relative** (0.7× each task's own specialist); absolute normalized retention is what the table reports.
- Difficulty is from dynamics (slip/obstacles/goals); **reward magnitude is uniform** [0,1] to keep returns O(1) for a single eps.
- No figures (per instruction). Raw per-run artifacts under reports/gridworld/<run>/; full metrics in metrics.json.
