#!/bin/bash
# The experiment the plan's gate actually asks for, and that step 2 skipped.
#
# Every step-2 cell ran the SAME schedule (axial 3000 -> 800 -> 3000, lateral
# 1500 -> 300, K_R 80 -> 20) and varied its levels.  Comparing levels of a
# schedule to each other cannot show whether the schedule is needed.  This runs
# one stiffness for the whole episode against the scheduled baseline at the best
# levels found, so the comparison is schedule vs no-schedule.
#
# The axial level is the one with something real at stake: the search wants it
# soft (the press is k_axial * depth, and a soft spring slides over the sharp
# edge instead of wedging) while the push wants it stiff.  A single value has to
# serve both.  Press is held at 4 N throughout by deriving the depth.
#SBATCH --job-name=peg_sch
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/peg3/%x_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CELL=(
  "SCHED_e4    --errors 4"
  "SCHED_e8    --errors 8"
  "CONST800_e4  --no-schedule --k-axial-search 800  --errors 4"
  "CONST800_e8  --no-schedule --k-axial-search 800  --errors 8"
  "CONST1600_e4 --no-schedule --k-axial-search 1600 --errors 4"
  "CONST1600_e8 --no-schedule --k-axial-search 1600 --errors 8"
  "CONST3000_e4 --no-schedule --k-axial-search 3000 --errors 4"
  "CONST3000_e8 --no-schedule --k-axial-search 3000 --errors 8"
  "CONST400_e4  --no-schedule --k-axial-search 400  --errors 4"
  "CONST400_e8  --no-schedule --k-axial-search 400  --errors 8"
)
row=${CELL[$SLURM_ARRAY_TASK_ID]}
echo "### cell: ${row%% *}  [${row#* }]"
python -u peg_insertion_case1.py --episodes "${EPS:-8}" --press 4 --catch-frac 0.4 \
    --kr-search 20 --k-lat 300 ${row#* }
