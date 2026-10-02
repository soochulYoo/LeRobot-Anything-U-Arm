#!/bin/bash
# Closed-loop evaluation of every act variant, all seeds.
#   bash slurm/submit_eval.sh            10 episodes per text, per seed
#   EPISODES=20 bash slurm/submit_eval.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs
for n in act_ft_input act_uni_dir act_unified act_cross_cond act_film act_ft_tokens; do
  [ -d "runs/$n" ] || { echo "skip $n (not trained)"; continue; }
  NAME="$n" sbatch -J "wipeval_$n" --export=ALL slurm/eval_act.sbatch
done
