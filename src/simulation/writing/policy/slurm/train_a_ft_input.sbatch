#!/bin/bash
#SBATCH --job-name=writing_a_ft_input
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (a) ft_input -- current force is only an input to the action flow (baseline)
#
# One array task per seed (0-2), each writing policy/runs/ft_input/seed<k>/model.pkl.
# Submit from src/simulation/writing:
#   mkdir -p logs && sbatch policy/slurm/train_a_ft_input.sbatch
# --time is a guess: ~40 ms/step on a CPU gives 20k steps in ~15 min, and a GPU
# should be well under that.  Check logs/ and adjust.

export CASE=a NAME=ft_input
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
