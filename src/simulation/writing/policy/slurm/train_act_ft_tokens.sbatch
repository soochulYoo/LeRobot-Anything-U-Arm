#!/bin/bash
#SBATCH --job-name=writing_act_ft_tokens
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (a) ft_input with the measured F/T history as 10 tokens with frame positions, so the action queries can re-read a moment of the trace instead of one GRU summary.
#
# See policy/FORCE_CONDITIONING.md for the four mechanisms this belongs to.
# Parameter-matched to the other act structures, so the mechanism is the only
# thing that differs.

export CASE=a NAME=act_ft_tokens ARCH=act
export EXTRA_ARGS="--match-params --ft-tokens ${EXTRA_ARGS:-}"
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
