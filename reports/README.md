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

`<method>` is one of `constrained` (ours, full theory / constrained-local),
`localfree` (ours, unconstrained-local variant), `finetune` (naive single-network
sequential baseline).

## Current experiments

- `minatar_theory/` — four MinAtar games (SpaceInvaders → Breakout → Asterix →
  Seaquest), shared conv trunk + per-task heads, pure REINFORCE, 10 seeds.
  Compares constrained-local vs unconstrained-local vs naive fine-tuning.

## Regenerating

```bash
# raw runs (per seed, 3 method-runs):
sbatch scripts/hpc_minatar.sbatch constrained <seed> configs/minatar_multihead.yaml minatar_multihead
sbatch scripts/hpc_minatar.sbatch finetune    <seed> configs/minatar_multihead.yaml minatar_multihead
sbatch scripts/hpc_minatar.sbatch constrained <seed> configs/minatar_localfree.yaml minatar_localfree
# curated CI bundle:
python -m experiments.aggregate_theory --seeds 0 1 2 3 4 5 6 7 8 9
```
