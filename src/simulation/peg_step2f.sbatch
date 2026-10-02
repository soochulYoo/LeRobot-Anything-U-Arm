#!/bin/bash
# Step 2, final: where does the winning constant BREAK?
#
# The disambiguation settled it -- with the catch threshold held fixed, a 4 N
# press inserts 4/4 at every aim error tested, while 8/16/24 N do not.  So the
# press is the knob and the catch was not the explanation.  But 4/4 on 4 seeds
# up to 8 mm is not the same as "nothing left to learn", and the §8 gate turns
# on exactly that.  Two things are missing:
#   8 seeds instead of 4, so a 4/4 is not luck
#   errors past 8 mm, until it fails
# Note the spiral only reaches 1 + 12 = 13 mm, so beyond ~13 mm the search MUST
# fail for a geometric reason that has nothing to do with stiffness.  The last
# three cells grow the spiral to 25 mm so the ceiling measured is the task's and
# not the script's.
#SBATCH --job-name=peg_s2f
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-11
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/peg2/%x_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CELL=(
  "F_p2_e4    --press 2  --errors 4"
  "F_p2_e8    --press 2  --errors 8"
  "F_p2_e12   --press 2  --errors 12"
  "F_p4_e4    --press 4  --errors 4"
  "F_p4_e8    --press 4  --errors 8"
  "F_p4_e12   --press 4  --errors 12"
  "F_p8_e4    --press 8  --errors 4"
  "F_p8_e8    --press 8  --errors 8"
  "F_p8_e12   --press 8  --errors 12"
  "G_wide_e12 --press 4  --errors 12 --search-growth 0.024 --search-turns 9"
  "G_wide_e18 --press 4  --errors 18 --search-growth 0.024 --search-turns 9"
  "G_wide_e24 --press 4  --errors 24 --search-growth 0.024 --search-turns 9"
)
row=${CELL[$SLURM_ARRAY_TASK_ID]}
echo "### cell: ${row%% *}  [${row#* }]"
python -u peg_insertion_case1.py --episodes "${EPS:-8}" --k-axial-search 800 \
    --search-catch 0.005 ${row#* }
