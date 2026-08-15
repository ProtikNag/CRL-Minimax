#!/bin/bash
# ── Full brute-force Eq-29 consolidation run (4 games, per-iteration values) ──
#   sbatch scripts/hpc_brute.sbatch [config] [seed]
#SBATCH --job-name=crl-brute
#SBATCH --partition=gpu,gpu-v100-16gb,gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-brute-%j.out
#SBATCH --error=slurm-brute-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
set -euo pipefail
cd /work/pnag/CRL-Minimax
CONFIG=${1:-configs/consolidate4_brute.yaml}
SEED=${2:-0}
module load python3/anaconda/2023.7 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate claudecli
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) config=$CONFIG seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -u -m experiments.run --config "$CONFIG" --seed "$SEED"
