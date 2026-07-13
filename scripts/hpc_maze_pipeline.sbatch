#!/bin/bash
# ── Autonomous driver for the continual-maze study ──
# Picks the maze size (learnability), writes configs, submits 30 worker jobs
# (10 seeds x 3 methods), then submits an aggregation job that runs after them.
# Fully self-contained: submit this one job and everything else follows.
#SBATCH --job-name=crl-maze-drv
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-maze-drv-%j.out
#SBATCH --error=slurm-maze-drv-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -uo pipefail
cd /work/pnag/CRL-Minimax
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "[drv] host=$(hostname) start $(date)"

# 1. pick maze size + write configs
python -m experiments.maze_pick_and_configure
if [ ! -f configs/maze_20task.yaml ]; then echo "[drv] config generation FAILED"; exit 1; fi

# 2. submit 30 workers (10 seeds x {constrained, finetune, localfree})
IDS=""
for seed in 0 1 2 3 4 5 6 7 8 9; do
  A=$(sbatch --parsable scripts/hpc_maze_worker.sbatch constrained $seed configs/maze_20task.yaml maze_20task)
  B=$(sbatch --parsable scripts/hpc_maze_worker.sbatch finetune    $seed configs/maze_20task.yaml maze_20task)
  C=$(sbatch --parsable scripts/hpc_maze_worker.sbatch constrained $seed configs/maze_20task_localfree.yaml maze_20task_localfree)
  IDS="$IDS:$A:$B:$C"
done
IDS=${IDS#:}
echo "[drv] submitted 30 workers: $IDS"

# 3. submit aggregation, dependent on all workers finishing (afterany = tolerate failures)
AGG=$(sbatch --parsable --dependency=afterany:$IDS scripts/hpc_maze_aggregate.sbatch)
echo "[drv] submitted aggregator $AGG (afterany all workers)"
echo "[drv] done $(date)"
