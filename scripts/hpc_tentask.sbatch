#!/bin/bash
# ── 10-task, 5-seed proof-of-concept (real rollouts) as a SLURM array job ──
#
# One array task per seed runs all four methods (constrained / unconstrained /
# finetune / joint) for the sampled 10-task gridworld. CPU-only: the gridworld
# tiers need no GPU, and the vectorized rollout makes each seed ~15-20 min.
#
# Submit the array, then aggregate once it finishes:
#   sbatch scripts/hpc_tentask.sbatch
#   # after all 5 array tasks complete:
#   python -m experiments.aggregate_seeds \
#       --config configs/gridworld_tentask_sampled.yaml \
#       --name gridworld_tentask --seeds 0 1 2 3 4
#
# (or submit the companion aggregation job with a dependency -- see the tail.)

#SBATCH --job-name=crl-tentask
#SBATCH --partition=defq-64core
#SBATCH --array=0-4
#SBATCH --output=slurm-tentask-%A_%a.out
#SBATCH --error=slurm-tentask-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG=configs/gridworld_tentask_sampled.yaml
NAME=gridworld_tentask
SEED=${SLURM_ARRAY_TASK_ID:-0}

# ── Environment (mirror the working login-node setup) ──────────────────────
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "host=$(hostname) date=$(date -u +%FT%TZ) seed=$SEED commit=$(git rev-parse --short HEAD)"
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"

# ── Run this seed's four methods (raw output -> results/, gitignored) ───────
python -m experiments.multiseed_comparison \
    --config "$CONFIG" --name "$NAME" --seed "$SEED"

echo "seed $SEED done. Aggregate after all seeds finish:"
echo "  python -m experiments.aggregate_seeds --config $CONFIG --name $NAME --seeds 0 1 2 3 4"
