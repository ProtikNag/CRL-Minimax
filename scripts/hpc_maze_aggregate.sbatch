#!/bin/bash
# ── Aggregate the 20-maze study + commit/push (runs after all workers) ──
#SBATCH --job-name=crl-maze-agg
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-maze-agg-%j.out
#SBATCH --error=slurm-maze-agg-%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -uo pipefail
cd /work/pnag/CRL-Minimax
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) aggregating maze_20task at $(date)"

python -m experiments.aggregate_theory \
    --name maze_20task --config configs/maze_20task.yaml \
    --constrained-name maze_20task --localfree-name maze_20task_localfree \
    --seeds 0 1 2 3 4 5 6 7 8 9 --perf-episodes 30

git add reports/maze_20task configs/maze_20task.yaml configs/maze_20task_localfree.yaml
git -c user.name=ProtikNag -c user.email=protiknag08@gmail.com commit -m "Continual maze (20 mazes, 10 seeds): 3-method results + visualizations

Autonomous SLURM pipeline result. One maze per task (walls, dead-ends, braided
loops), start/goal random per episode (BFS-reachable), per-maze wall-hit penalty,
sampled REINFORCE, one shared head, no task-id. Bundle: retention/learning/reward
curves, forgetting matrices, success + value tables, average-performance curve.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "[agg] nothing to commit"
git push || echo "[agg] push failed (no network on compute node?); committed locally"
echo "[agg] done at $(date)"
