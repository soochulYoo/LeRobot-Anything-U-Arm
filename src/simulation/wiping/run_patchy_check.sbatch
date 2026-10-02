#!/bin/bash
#SBATCH --job-name=patchy_check
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WIPE_PAD_R=0.040 WIPE_SURFACE=patchy
# does a constant K_R still win when the traverse crosses both regimes?
python -u oracle_kr.py --board curved --kr 60,10,3,1,0.3 --tol 0,2 --text "${TEXT:-S}" --letter "${LETTER:-0.15}" ${EXTRA:-}
