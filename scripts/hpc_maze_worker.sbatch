#!/bin/bash
# ── One continual-maze run: one method, one seed ──
#   sbatch scripts/hpc_maze_worker.sbatch <method> <seed> <config> <name>
#SBATCH --job-name=crl-maze
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-maze-%j.out
#SBATCH --error=slurm-maze-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
METHOD=${1:?method}; SEED=${2:?seed}; CONFIG=${3:?config}; NAME=${4:?name}
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) method=$METHOD seed=$SEED config=$CONFIG name=$NAME commit=$(git rev-parse --short HEAD)"
python -m experiments.multiseed_comparison --config "$CONFIG" \
    --name "$NAME" --seed "$SEED" --methods "$METHOD"
