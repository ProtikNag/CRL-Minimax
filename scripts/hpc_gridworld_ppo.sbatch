#!/bin/bash
# ── Hard 50-task GridWorld, PPO actor-critic backend, ONE seed on ONE GPU ──
#   sbatch scripts/hpc_gridworld_ppo.sbatch <seed> [config]
# Submit one job per seed to run 3 seeds on 3 separate GPUs concurrently.
# NOTE: the workload is tiny-tensor + Python gym stepping, so it is CPU/env-bound;
# OMP_NUM_THREADS=1 is REQUIRED (multithreading thrashes on trivial ops and ~2x's
# the per-iter time). The GPU gives per-seed isolation + parallelism, not per-iter
# speedup.
#SBATCH --job-name=crl-grid50
#SBATCH --partition=dgx_aic
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-grid50-%j.out
#SBATCH --error=slurm-grid50-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euo pipefail
cd /work/pnag/CRL-Minimax
SEED=${1:?usage: sbatch hpc_gridworld_ppo.sbatch <seed> [config]}
CONFIG=${2:-configs/biggrid_50task.yaml}
export CLAUDE_CODE_TMPDIR=/work/pnag/.claude-tmp
export OMP_NUM_THREADS=1          # single-thread is ~2x faster here (tiny ops)
export MKL_NUM_THREADS=1
export PYTHONPATH=/work/pnag/CRL-Minimax
export PYTHONUNBUFFERED=1
echo "host=$(hostname) seed=$SEED config=$CONFIG commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
python -c "import torch; print('cuda', torch.cuda.is_available())"
python -m experiments.run --config "$CONFIG" --seed "$SEED"
echo "DONE gridworld50 seed=$SEED"
