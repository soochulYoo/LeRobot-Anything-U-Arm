#!/bin/bash
# Is "a constant is enough" a fact about insertion, or about THIS instance?
#
# The task ships a 3 mm gap, which on a 15-25 mm half-width peg is 12-20%.
# Precision assembly runs 0.1-1%.  This retests schedule-vs-constant at 2 mm and
# 1 mm of clearance, where the multi-contact wedging that motivates impedance
# scheduling actually has room to happen.
#
# THE SPIRAL HAS TO BE RESCALED, or tight clearance fails for a scripted reason.
# The head drops in only if it passes within `clearance` of the hole centre, so a
# spiral whose radial pitch exceeds that steps straight over the opening.  At
# 3 mm the pitch was 12/5 = 2.4 mm, just inside the window; at 1 mm it would be
# 2.4x too coarse and every failure would be the script's, not the physics'.  So
# the pitch is held at the clearance (turns = growth / clearance) and search_s is
# stretched to keep the TANGENTIAL SPEED at the 3 mm baseline's 0.055 m/s --
# widening the spiral at fixed time is what made 12 mm worse, so speed is the
# thing that must not drift.
#SBATCH --job-name=peg_cl
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=logs/peg4/%x_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ARM=("SCHED" "CONST400" "CONST800" "CONST1600" "CONST3000")
FLAGS=("" "--no-schedule --k-axial-search 400" "--no-schedule --k-axial-search 800" \
       "--no-schedule --k-axial-search 1600" "--no-schedule --k-axial-search 3000")
i=$SLURM_ARRAY_TASK_ID
c=$(( i / 5 ))            # 0 -> 2 mm, 1 -> 1 mm
a=$(( i % 5 ))
if [ "$c" -eq 0 ]; then CL=2; TURNS=6;  SS=5;  DUR=20; else CL=1; TURNS=12; SS=10; DUR=25; fi
echo "### cell: ${ARM[$a]} at ${CL} mm clearance  [${FLAGS[$a]}]"
python -u peg_insertion_case1.py --episodes "${EPS:-8}" --errors "${ERRS:-0,2,4,6}" \
    --clearance-mm "$CL" --search-turns "$TURNS" --search-s "$SS" --duration "$DUR" \
    --press 4 --catch-frac 0.4 --kr-search 20 --k-lat 300 ${FLAGS[$a]}
