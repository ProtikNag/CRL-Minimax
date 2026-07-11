#!/bin/bash
# ── Seed-averaged aggregation for the CartPole study (run after the array) ──
#
#   sbatch --dependency=afterok:$ARRAY scripts/hpc_cartpole_aggregate.sbatch

#SBATCH --job-name=crl-cartpole-agg
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-cartpole-agg-%j.out
#SBATCH --error=slurm-cartpole-agg-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG=configs/cartpole_multihead.yaml
NAME=cartpole_multihead

module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m experiments.aggregate_seeds \
    --config "$CONFIG" --name "$NAME" --seeds 0 1 2 \
    --methods constrained finetune joint --perf-episodes 100

echo "aggregation done -> reports/$NAME/figures/{png,svg}/ and tables/"
