# Reports (committed experiment bundles)

Curated, shareable outputs. This directory **is tracked in git** so results can
be committed on HPC and pulled locally. Raw runs (logs, checkpoints) live under
`results/` at the repo root and are **gitignored** — they regenerate on any
machine from the configs.

## Layout

```
reports/<experiment_name>/
├── figures/
│   ├── png/                     # cross-method figures (retention curves, bars,
│   ├── svg/                     #   design-space map, method schematic, tables)
│   └── <method>/{png,svg}/      # per-method diagnostics (learning curves,
│                                #   duals, gaps, forgetting matrix)
├── tables/
│   ├── retention_table.csv      # final value + % of expert retained, per task
│   └── performance_table.csv    # success rate / return / steps (real rollouts)
└── eval_matrix_<method>.json    # raw task×phase evaluation matrices
```

`<method>` is one of `constrained` (ours), `finetune` (naive single-network
sequential), `unconstrained` (constraint-off ablation), `joint` (upper bound).

## Current experiments

- `nn_three_task/` — 3-task 5×5 gridworld, multi-head MLP, exact estimator.
  The first neural multi-task result.
- `gridworld_manytask/` — 6-task 7×7 gridworld, exact. Scaling check: ours
  retains all six; fine-tuning shows decaying catastrophic forgetting.

## Regenerating

```bash
python -m experiments.baseline_comparison \
    --config configs/<cfg>.yaml --name <experiment_name>
```

Writes both the raw runs (to `results/`) and this curated bundle (to
`reports/<experiment_name>/`).
