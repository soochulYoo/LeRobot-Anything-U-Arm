#!/bin/bash
#SBATCH --job-name=rot_stiff
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
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
python -u record.py --hi "${HI:-60}" --lo "${LO:-3}" --out "${OUT:-rot_stiffness}"
