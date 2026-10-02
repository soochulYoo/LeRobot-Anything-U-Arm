#!/bin/bash
# Which seed shows the contrast?  The step-1 table aggregates 4 seeds, so it says
# 2/4 jammed at 4 mm without saying which.  A video needs the specific episode
# where aim-and-push jams and the scrub goes in.
#SBATCH --job-name=peg_pick
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --output=logs/peg1/pick_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
E=(4 4 6 6); S=(4 0 4 0)
e=${E[$SLURM_ARRAY_TASK_ID]}; ss=${S[$SLURM_ARRAY_TASK_ID]}
python -u peg_insertion_case1.py --errors "$e" --episodes 4 --search-s "$ss" --per-seed
