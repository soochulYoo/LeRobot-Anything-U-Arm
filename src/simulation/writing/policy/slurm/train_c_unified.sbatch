#!/bin/bash
#SBATCH --job-name=writing_c_unified
#SBATCH --partition=suma_a6000
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (c) unified -- one flow over [action | future force], one flow time
#
# One array task per seed (0-2), each writing policy/runs/unified/seed<k>/model.pkl.
# Submit from src/simulation/writing:
#   mkdir -p logs && sbatch policy/slurm/train_c_unified.sbatch
# --time is a guess: ~40 ms/step on a CPU gives 20k steps in ~15 min, and a GPU
# should be well under that.  Check logs/ and adjust.

export CASE=c NAME=unified
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
