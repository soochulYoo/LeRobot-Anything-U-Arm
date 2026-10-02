#!/bin/bash
#SBATCH --job-name=wipe_dataset
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export WIPE_PAD_R="${PAD:-0.040}"
python -u surface_dataset.py "${ROOT:-demos/wipe_curved}" --out "${OUT:-data/surface.npz}"
