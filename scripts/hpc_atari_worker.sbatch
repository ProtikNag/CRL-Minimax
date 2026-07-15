#!/bin/bash
# ── One Atari PPO continual run: one config (method), one seed ──
#   sbatch scripts/hpc_atari_worker.sbatch <config> <seed>
# The run directory name (results/<experiment.name>_seed<seed>) is taken from the
# config's experiment.name, which encodes the method (constrained / finetune).
#SBATCH --job-name=crl-atari
#SBATCH --partition=gpu,gpu-v100-16gb,gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-atari-%j.out
#SBATCH --error=slurm-atari-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
CONFIG=${1:?config}; SEED=${2:?seed}
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) config=$CONFIG seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m experiments.run --config "$CONFIG" --seed "$SEED"
