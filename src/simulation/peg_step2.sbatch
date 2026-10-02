#!/bin/bash
# Step 2 of PEG_DEMO_PLAN.md: does ONE constant stiffness do the whole job?
# Four sub-sweeps, because Case 1's knobs are coupled and one grid would
# confound them:
#   A  axial stiffness at a HELD press (depth = press/k_axial, catch = 40% of it)
#   B  press at a held axial stiffness
#   C  lateral stiffness
#   D  K_R
#SBATCH --job-name=peg_s2
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/peg2/%x_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CELL=(
  "A_kax400   --press 8 --k-axial-search 400"
  "A_kax800   --press 8 --k-axial-search 800"
  "A_kax1600  --press 8 --k-axial-search 1600"
  "A_kax3000  --press 8 --k-axial-search 3000"
  "B_press4   --press 4 --k-axial-search 800"
  "B_press8   --press 8 --k-axial-search 800"
  "B_press16  --press 16 --k-axial-search 800"
  "B_press24  --press 24 --k-axial-search 800"
  "C_klat150  --press 8 --k-lat 150"
  "C_klat300  --press 8 --k-lat 300"
  "C_klat800  --press 8 --k-lat 800"
  "C_klat2000 --press 8 --k-lat 2000"
  "D_kr2      --press 8 --kr-search 2"
  "D_kr10     --press 8 --kr-search 10"
  "D_kr40     --press 8 --kr-search 40"
  "D_kr80     --press 8 --kr-search 80"
)
row=${CELL[$SLURM_ARRAY_TASK_ID]}
label=${row%% *}
flags=${row#* }
echo "### cell: $label  [$flags]"
python -u peg_insertion_case1.py --errors "${ERRS:-0,4,8}" \
    --episodes "${EPS:-4}" --catch-frac 0.4 $flags
