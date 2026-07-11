#!/bin/bash
# ── CartPole continual study: 3-seed SLURM array (constrained/finetune/joint) ──
#
# One array task per seed runs the three methods on the 6-task CartPole family.
# CPU-only; the batched torch CartPole rollout keeps it affordable.
#
#   ARRAY=$(sbatch --parsable scripts/hpc_cartpole.sbatch)
#   sbatch --dependency=afterok:$ARRAY scripts/hpc_cartpole_aggregate.sbatch

#SBATCH --job-name=crl-cartpole
#SBATCH --partition=defq-64core
#SBATCH --array=0-2
#SBATCH --output=slurm-cartpole-%A_%a.out
#SBATCH --error=slurm-cartpole-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG=configs/cartpole_multihead.yaml
NAME=cartpole_multihead
SEED=${SLURM_ARRAY_TASK_ID:-0}
METHODS="constrained finetune joint"

module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname) date=$(date -u +%FT%TZ) seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import torch; print('torch', torch.__version__)"

python -m experiments.multiseed_comparison \
    --config "$CONFIG" --name "$NAME" --seed "$SEED" --methods $METHODS

echo "seed $SEED done. Aggregate after all seeds finish:"
echo "  python -m experiments.aggregate_seeds --config $CONFIG --name $NAME \\"
echo "      --seeds 0 1 2 --methods $METHODS"
