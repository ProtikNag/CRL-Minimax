#!/bin/bash
# ── SLURM batch script for the many-task rollout experiment (Hyperion / HPC) ──
#
# Runs the four-method baseline comparison on the sampled many-task config
# (real environment rollouts). Edit the SBATCH lines for your cluster's
# partition / account, then:   sbatch scripts/hpc_baseline.sbatch
#
# Placeholders marked <...> must be set for your cluster.

#SBATCH --job-name=crl-minimax
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# #SBATCH --partition=<your-partition>
# #SBATCH --account=<your-account>
# #SBATCH --gres=gpu:1            # not needed for the gridworld/CartPole tiers

set -euo pipefail

# 1) Environment. Prefer a project venv; create it once, reuse thereafter.
module load python 2>/dev/null || true          # adjust to your module system
if [ ! -d .venv ]; then
    python -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
fi
PY=./.venv/bin/python

# 2) Run header (reproducibility): record what is about to run.
echo "host=$(hostname) date=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD)"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 3) The experiment. Raw runs -> results/ (gitignored); figures+tables -> reports/.
CONFIG=configs/gridworld_manytask_sampled.yaml
NAME=gridworld_manytask_sampled
$PY -m experiments.baseline_comparison --config "$CONFIG" --name "$NAME"

# 4) Commit the curated report so it can be pulled locally.
#    (Only reports/ is tracked; results/ stays on the node.)
git add reports/"$NAME"
git commit -m "HPC run: $NAME (commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID)" || true
# git push        # uncomment once credentials are configured on the node

echo "done. figures in reports/$NAME/figures/{png,svg}/, tables in reports/$NAME/tables/"
