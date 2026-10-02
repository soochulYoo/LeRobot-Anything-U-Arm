#!/bin/bash
#SBATCH --job-name=wipe_collect
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
# The cameras need the GPU: an episode that renders at 10 Hz is minutes of CPU
# software rasterising on a login node and seconds here.  Everything else in
# this directory is physics only and runs anywhere.
# eval, so a shape case can be passed as TEXTS='S 7 "<star>"' -- unquoted,
# `<star>` is a shell redirect and the job dies before it starts.
eval "python -u protocol.py --board '${BOARD:-curved}' --texts ${TEXTS:-S 7} \
    --per-case '${PER_CASE:-50}' --auto-user --headless \
    --workers '${WORKERS:-8}' --video '${VIDEO:-2}' --out '${OUT:-demos/wiping}' \
    --kr-land '${KRLAND:-HIGH}'"
