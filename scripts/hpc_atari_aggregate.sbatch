#!/bin/bash
# ── Aggregate Atari PPO runs into figures/tables (+ commit) ──
#   sbatch scripts/hpc_atari_aggregate.sbatch "<seed list>"   e.g. "0 1 2"
#SBATCH --job-name=crl-atari-agg
#SBATCH --partition=gpu,gpu-v100-16gb,gpu-v100-32gb
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-atari-agg-%j.out
#SBATCH --error=slurm-atari-agg-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -uo pipefail
cd /work/pnag/CRL-Minimax
SEEDS=${1:-0}
module load python3/anaconda/2023.7 2>/dev/null || true
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "[agg] host=$(hostname) seeds=$SEEDS start $(date)"
python -m experiments.aggregate_atari --name atari5_ppo --seeds $SEEDS

# Commit figures/tables + the small eval matrices (not the large policy blobs).
git add -f reports/atari5_ppo results/atari5_ppo_*_seed*/eval_matrix.json 2>/dev/null || true
if ! git diff --cached --quiet; then
  git -c user.name='ProtikNag' -c user.email='protiknag08@gmail.com' \
      commit -m "Atari5 PPO continual results: figures + tables (seeds $SEEDS)" || true
  git push origin HEAD:main || echo "[agg] push failed (push manually)"
else
  echo "[agg] nothing to commit"
fi
echo "[agg] done $(date)"
