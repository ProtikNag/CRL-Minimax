#!/bin/bash
#SBATCH --job-name=box10
#SBATCH --partition=dgx_aic_short,gpu,gpu-v100-32gb,gpu-v100-16gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-box10-%j.out
#SBATCH --error=slurm-box10-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export PYTHONPATH=/work/pnag/CRL-Minimax
export BOX_ITERS=10
echo "host=$(hostname) BOX_ITERS=$BOX_ITERS commit=$(git rev-parse --short HEAD)"
python -c "import torch;print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m experiments.boxing_topup
