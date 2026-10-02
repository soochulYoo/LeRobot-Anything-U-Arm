#!/bin/bash
# Rendering needs a GPU, unlike every other peg job here.
#
# Three cells, chosen from the per-seed probe (logs/peg1/pick_*.log) rather than
# picked for looks:
#   4 mm seed 0   push JAMS at 10.7 mm / 52 N, scrub goes in at 25 N
#   6 mm seed 0   push NEVER FINDS IT, scrub goes in
#   6 mm seed 2   the COUNTEREXAMPLE: push jams at 8.2 mm but the scrub misses
#                 entirely.  The search is not uniformly better and the video
#                 set should say so.
#SBATCH --job-name=peg_vid
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=logs/peg1/vid_%a.log
#SBATCH --array=0-2
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
E=(4 6 6); S=(0 0 2); TAG=(4mm 6mm 6mm_counterexample)
i=$SLURM_ARRAY_TASK_ID
python -u peg_record_search.py --err "${E[$i]}" --seed "${S[$i]}" \
    --out "videos/peg_search_${TAG[$i]}"
