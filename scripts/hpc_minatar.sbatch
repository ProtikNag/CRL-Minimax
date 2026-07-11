#!/bin/bash
# ── MinAtar continual run: one method, one seed (CPU; run methods in parallel) ──
#   sbatch scripts/hpc_minatar.sbatch <method> [seed]
#SBATCH --job-name=crl-minatar
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-minatar-%j.out
#SBATCH --error=slurm-minatar-%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
cd /work/pnag/CRL-Minimax
METHOD=${1:?usage: sbatch hpc_minatar.sbatch <method> [seed]}
SEED=${2:-0}
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) method=$METHOD seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import minatar, torch; print('minatar OK, torch', torch.__version__)"
python -m experiments.multiseed_comparison --config configs/minatar_multihead.yaml \
    --name minatar_multihead --seed "$SEED" --methods "$METHOD"
