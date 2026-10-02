#!/bin/bash
#SBATCH --job-name=wipe_oracle
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export WIPE_PAD_R="${PAD:-0.015}"
echo "pad half-width ${WIPE_PAD_R} m"
python -u oracle_kr.py --board "${BOARD:-curved}" --kr "${KR:-60,30,10,3,1,0.3}" ${EXTRA:-}
