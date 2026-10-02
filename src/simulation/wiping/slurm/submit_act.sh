#!/bin/bash
# The six act variants of the wiping policy, three seeds each.
#
#   bash slurm/submit_act.sh                      the approved protocol's demos
#   DATA=$PWD/demos/wipe_curved_krlow bash slurm/submit_act.sh      the A/B set
#   STEPS=40000 bash slurm/submit_act.sh
#
# Variants are ../writing/policy/ARCHITECTURE.md's: a is Comp-ACT unmodified,
# b adds a future-force predictor, c one head over [action | force], d the
# cross-conditioned pair, e the wrench through adaLN-Zero, and ft_tokens is a
# with the wrench as ten tokens instead of one GRU summary.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs

submit() {   # name case [extra args]
  NAME="$1" CASE="$2" EXTRA_ARGS="${3:-}" \
    sbatch -J "wipe_$1" --export=ALL slurm/train_act.sbatch
}

submit act_ft_input   a
submit act_uni_dir    b
submit act_unified    c
submit act_cross_cond d
submit act_film       e "--match-params"
submit act_ft_tokens  a "--match-params --ft-tokens"
