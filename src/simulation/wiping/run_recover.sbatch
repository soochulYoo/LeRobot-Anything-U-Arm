#!/bin/bash
#SBATCH --job-name=wipe_recover
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WIPE_PAD_R="${PAD:-0.040}"
python -u recover.py --kr "${KR:-60,3,0.3}" ${EXTRA:-}
