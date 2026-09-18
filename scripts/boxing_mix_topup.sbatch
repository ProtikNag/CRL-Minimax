#!/bin/bash
#SBATCH --job-name=boxmix
#SBATCH --partition=dgx_aic_short,gpu,gpu-v100-32gb,gpu-v100-16gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-boxmix-%j.out
#SBATCH --error=slurm-boxmix-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
python -c "import torch;print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m experiments.boxing_mix_topup
