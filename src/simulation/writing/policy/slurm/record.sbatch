#!/bin/bash
#SBATCH --job-name=writing_record
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

# One episode as a video: writing, force and stiffness on one canvas.
# The GPU is for SAPIEN's renderer.
#   CKPT  checkpoint        TEXT  S | 7 | <star>        OUT  mp4 path
set -euo pipefail
cd "${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"
python -u policy/record_episode.py --ckpt "${CKPT:?}" --text "${TEXT:-S}" \
  ${OUT:+--out "$OUT"} ${EXTRA_ARGS:-}
