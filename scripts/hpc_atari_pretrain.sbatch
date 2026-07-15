#!/bin/bash
# ── Per-game local-PPO sanity check (verify each game learns a decent score) ──
#   sbatch scripts/hpc_atari_pretrain.sbatch <Game> [iters]
#SBATCH --job-name=crl-atari-pre
#SBATCH --partition=gpu,gpu-v100-16gb,gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-atari-pre-%j.out
#SBATCH --error=slurm-atari-pre-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
GAME=${1:?game}; ITERS=${2:-600}
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) game=$GAME iters=$ITERS commit=$(git rev-parse --short HEAD)"
python -m experiments.atari_pretrain_check --game "$GAME" --iters "$ITERS" \
    --eval-every 50 --device auto
