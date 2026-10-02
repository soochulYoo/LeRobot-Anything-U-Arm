#!/bin/bash
#SBATCH --job-name=wipe_act
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.log
#
# One act variant of the wiping policy, three seeds (the array index).
#
# The body is ../writing/policy/slurm/_train.sh, unchanged: the policy code is
# shared, and the only thing that makes this wiping is the dataset -- whose
# action carries a rotational stiffness, so `train.py` builds a 7-wide action
# head instead of 6 and scores K_R against the protocol's own ladder.
#
# Submit through slurm/submit_act.sh, which sets NAME and CASE per variant.
#   NAME   run directory under wiping/runs/
#   CASE   a | b | c | d | e   the structure (see ../writing/policy/ARCHITECTURE.md)
#   DATA   default wiping/demos/wipe_curved; the --kr-land A/B set is
#          demos/wipe_curved_krlow and the flat one demos/wipe_flat
set -euo pipefail
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
export WRITING_DIR="${WRITING_DIR:-$(cd "$HERE/../writing" && pwd)}"
export ARCH=act
export DATA="${DATA:-$HERE/demos/wipe_curved}"
export CASE="${CASE:?set CASE}" NAME="${NAME:?set NAME}"
export SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export OUT="${OUT:-$HERE/runs/$NAME/seed$SEED}"
bash "$WRITING_DIR/policy/slurm/_train.sh"
