#!/bin/bash
# ── Autonomous driver for the Atari-5 PPO continual study ──
# Submits: (1) 5 per-game local-PPO sanity checks, (2) constrained + finetune
# continual runs per seed on the GPU partition, (3) an aggregation job that runs
# after the continual runs finish and commits the figures/tables.
#   sbatch scripts/hpc_atari_pipeline.sbatch ["0 1 2"]
#SBATCH --job-name=crl-atari-drv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-atari-drv-%j.out
#SBATCH --error=slurm-atari-drv-%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
set -uo pipefail
cd /work/pnag/CRL-Minimax
SEEDS=${1:-0}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "[drv] host=$(hostname) seeds=$SEEDS start $(date)"

# 1. per-game sanity checks (independent; verify each game learns)
for g in Pong Breakout Boxing Freeway SpaceInvaders; do
  sbatch scripts/hpc_atari_pretrain.sbatch "$g" 600 || true
done

# 2. continual runs: constrained + finetune per seed
IDS=""
for s in $SEEDS; do
  A=$(sbatch --parsable scripts/hpc_atari_worker.sbatch configs/atari5_ppo.yaml "$s")
  B=$(sbatch --parsable scripts/hpc_atari_worker.sbatch configs/atari5_ppo_finetune.yaml "$s")
  IDS="$IDS:$A:$B"
  echo "[drv] seed $s -> constrained=$A finetune=$B"
done
IDS=${IDS#:}

# 3. aggregation after all continual runs (afterany tolerates failures)
AGG=$(sbatch --parsable --dependency=afterany:$IDS scripts/hpc_atari_aggregate.sbatch "$SEEDS")
echo "[drv] submitted aggregator $AGG (afterany $IDS)"
echo "[drv] done $(date)"
