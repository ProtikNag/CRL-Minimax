#!/bin/bash
# ── Seed-averaged aggregation for the 10-task study (run after the array) ──
#
# Submit with a dependency on the array job so it fires automatically once all
# seeds finish:
#   ARRAY=$(sbatch --parsable scripts/hpc_tentask.sbatch)
#   sbatch --dependency=afterok:$ARRAY scripts/hpc_tentask_aggregate.sbatch

#SBATCH --job-name=crl-tentask-agg
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-tentask-agg-%j.out
#SBATCH --error=slurm-tentask-agg-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG=configs/gridworld_tentask_sampled.yaml
NAME=gridworld_tentask

module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m experiments.aggregate_seeds \
    --config "$CONFIG" --name "$NAME" --seeds 0 1 2 3 4

echo "aggregation done -> reports/$NAME/figures/{png,svg}/ and tables/"
