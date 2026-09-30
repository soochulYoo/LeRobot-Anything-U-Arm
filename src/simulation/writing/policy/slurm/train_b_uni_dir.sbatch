#!/bin/bash
#SBATCH --job-name=writing_b_uni_dir
#SBATCH --partition=suma_a6000
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log

# (b) uni_dir -- a future-force predictor (from the observation) feeds the action flow
#
# One array task per seed (0-2), each writing policy/runs/uni_dir/seed<k>/model.pkl.
# Submit from src/simulation/writing:
#   mkdir -p logs && sbatch policy/slurm/train_b_uni_dir.sbatch
# --time is a guess: ~40 ms/step on a CPU gives 20k steps in ~15 min, and a GPU
# should be well under that.  Check logs/ and adjust.

export CASE=b NAME=uni_dir
bash "${SLURM_SUBMIT_DIR:-$PWD}/policy/slurm/_train.sh"
