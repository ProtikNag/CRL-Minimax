#!/bin/bash
#SBATCH --job-name=box-topup
#SBATCH --partition=gpu,gpu-v100-32gb,gpu-v100-16gb,dgx_aic_short,AI_Center_L40S
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-boxtopup-%j.out
#SBATCH --error=slurm-boxtopup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /work/pnag/CRL-Minimax
module load python3/anaconda/2023.7 2>/dev/null || true
module load cuda 2>/dev/null || true
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
python -c "import torch;print('cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -m experiments.boxing_topup
