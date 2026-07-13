#!/bin/bash
# ── MinAtar Double-DQN min-max run (CPU) ──
#   sbatch scripts/hpc_minatar_dqn.sbatch <mode> [seed] [config]
#     mode: continual | experts
#SBATCH --job-name=crl-dqn
#SBATCH --partition=defq-64core
#SBATCH --output=slurm-dqn-%j.out
#SBATCH --error=slurm-dqn-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
cd /work/pnag/CRL-Minimax
MODE=${1:?usage: sbatch hpc_minatar_dqn.sbatch <mode:continual|experts> [seed] [config]}
SEED=${2:-0}
CONFIG=${3:-configs/minatar_dqn.yaml}
module load python3/anaconda/2023.7 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH=/work/pnag/CRL-Minimax
echo "host=$(hostname) mode=$MODE seed=$SEED config=$CONFIG commit=$(git rev-parse --short HEAD)"
python -c "import minatar, torch; print('minatar OK, torch', torch.__version__)"
python -m experiments.run_dqn --config "$CONFIG" --mode "$MODE" --seed "$SEED"
