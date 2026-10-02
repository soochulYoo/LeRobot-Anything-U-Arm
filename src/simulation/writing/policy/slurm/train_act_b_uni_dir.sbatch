#!/bin/bash
#SBATCH --job-name=writing_act_b_uni_dir
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (b) act_uni_dir -- a future-wrench decoder reads the observation alone and feeds the action decoder
#
# Comp-ACT: Kamijo, Beltran-Hernandez and Hamaya, arXiv:2406.14990.
# One array task per seed (0-2), each writing policy/runs/act_uni_dir/seed<k>/model.pkl.
# Submit from src/simulation/writing:
#   mkdir -p logs && sbatch policy/slurm/train_act_b_uni_dir.sbatch

export CASE=b NAME=act_uni_dir ARCH=act
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
