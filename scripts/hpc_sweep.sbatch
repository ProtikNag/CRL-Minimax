#!/bin/bash
# ── SLURM ARRAY job: sweep eps x dual-lr on the many-task exact tier ──
#
# Each array index runs one grid point (fast, exact estimator). Produces the
# constraint-strength ablation (paper Fig 4) across the grid.
#
#   sbatch --array=0-8 scripts/hpc_sweep.sbatch
# (grid size = product of the lists in configs/sweeps/eps_grid.yaml; 3x3 = 9)

#SBATCH --job-name=crl-sweep
#SBATCH --output=slurm-sweep-%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# #SBATCH --partition=<your-partition>

set -euo pipefail
PY=./.venv/bin/python
[ -d .venv ] || { python -m venv .venv && ./.venv/bin/pip install -r requirements.txt; }

$PY -m experiments.sweep \
    --base configs/gridworld_manytask_exact.yaml \
    --grid configs/sweeps/eps_grid.yaml \
    --seeds 0 1 2 \
    --index "${SLURM_ARRAY_TASK_ID}"
