#!/bin/bash
#SBATCH --job-name=writing_act_a_ft_input
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (a) act_ft_input -- Comp-ACT UNMODIFIED: the wrench is a decoder token, nothing predicts the future
#
# Comp-ACT: Kamijo, Beltran-Hernandez and Hamaya, arXiv:2406.14990.
# One array task per seed (0-2), each writing policy/runs/act_ft_input/seed<k>/model.pkl.
# Submit from src/simulation/writing:
#   mkdir -p logs && sbatch policy/slurm/train_act_a_ft_input.sbatch

export CASE=a NAME=act_ft_input ARCH=act
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
