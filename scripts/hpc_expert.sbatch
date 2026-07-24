#!/bin/bash
# ── Train one single-task Atari expert (Impala-CNN) to convergence ──
#   sbatch --partition=<part> scripts/hpc_expert.sbatch <Game> [seed]
# Writes everything to experts/<Game>/ (best_model, checkpoints, train_log.csv, meta.json).
#SBATCH --job-name=crl-expert
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-expert-%j.out
#SBATCH --error=slurm-expert-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
cd /work/pnag/CRL-Minimax
GAME=${1:?game}; SEED=${2:-0}
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) game=$GAME seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m experiments.train_expert --game "$GAME" --out "experts/$GAME" \
    --init experts/_shared_init.pt --seed "$SEED"
