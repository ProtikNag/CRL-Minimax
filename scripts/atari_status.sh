#!/bin/bash
# Quick status of the Atari PPO study: queue + latest pretrain scores + continual
# eval matrices. Usage: bash scripts/atari_status.sh
cd /work/pnag/CRL-Minimax
echo "===== QUEUE ($(date '+%H:%M:%S')) ====="
squeue -u pnag -o "%.10i %.16j %.8T %.10M %R" 2>/dev/null

echo; echo "===== PRETRAIN SCORES ====="
for g in Pong Breakout Boxing Freeway SpaceInvaders; do
  f=$(ls -t slurm-atari-pre-*.out 2>/dev/null | xargs grep -l "game=$g " 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    last=$(grep -E "score=|FINAL" "$f" 2>/dev/null | tail -1)
    printf "  %-14s %s\n" "$g" "${last:-<no score yet>}"
  else
    printf "  %-14s %s\n" "$g" "<no log>"
  fi
done

echo; echo "===== CONTINUAL RUNS ====="
for m in constrained finetune; do
  d="results/atari5_ppo_${m}_seed0"
  if [ -f "$d/eval_matrix.json" ]; then
    rows=$(python -c "import json;print(len(json.load(open('$d/eval_matrix.json'))))" 2>/dev/null)
    echo "  $m: eval_matrix rows=$rows (DONE if 5)"
  elif [ -d "$d" ]; then
    last=$(grep -vE "A.L.E|Stella" $d/../../slurm-atari-*.out 2>/dev/null | tail -0)
    echo "  $m: running (no eval_matrix yet)"
  else
    echo "  $m: not started"
  fi
done

echo; echo "===== RECENT CONTINUAL LOG TAILS ====="
for f in $(ls -t slurm-atari-[0-9]*.out 2>/dev/null | head -2); do
  echo "--- $f ---"; grep -vE "A.L.E|Stella" "$f" 2>/dev/null | tail -4
done

echo; echo "===== FAILURES ====="
grep -liE "Traceback|Error|CANCELLED|OOM|out of memory" slurm-atari-*.err 2>/dev/null | head