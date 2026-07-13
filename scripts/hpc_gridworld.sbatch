#!/bin/bash
# ── GridWorld continual run: one method, one seed (CPU, exact estimator) ──
#   sbatch scripts/hpc_gridworld.sbatch <method> [seed] [config] [name]
#SBATCH --job-name=crl-grid
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-grid-%j.out
#SBATCH --error=slurm-grid-%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
METHOD=${1:?usage: sbatch hpc_gridworld.sbatch <method> [seed] [config] [name]}
SEED=${2:-0}
CONFIG=${3:-configs/gridworld_20task.yaml}
NAME=${4:-gridworld_20task}
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) method=$METHOD seed=$SEED config=$CONFIG name=$NAME commit=$(git rev-parse --short HEAD)"
python -m experiments.multiseed_comparison --config "$CONFIG" \
    --name "$NAME" --seed "$SEED" --methods "$METHOD"
