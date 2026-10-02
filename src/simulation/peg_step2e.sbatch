#!/bin/bash
# Step 2, follow-up: sweep B confounded TWO knobs and I nearly reported it as one.
#
# --press derives search_depth = press/k_axial, and --catch-frac 0.4 then derives
# the catch threshold from that depth.  So "press 4 N" also meant "5 mm of
# reference depth and a 2 mm catch", while "press 24 N" meant "30 mm of depth and
# a 12 mm catch".  Sweep B's striking result -- 4 N inserts 4/4 at every aim
# error -- could just as well be "latch into the push as soon as the head is
# 2 mm in" as "press gently".  They are different claims with different
# consequences for the protocol, so they get separated here:
#   E0-E3  press varies, catch HELD at 2 mm
#   E4-E7  catch varies, press held at 4 and 8 N
#SBATCH --job-name=peg_s2e
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-7
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
  "E_p4_c2    --press 4  --search-catch 0.002"
  "E_p8_c2    --press 8  --search-catch 0.002"
  "E_p16_c2   --press 16 --search-catch 0.002"
  "E_p24_c2   --press 24 --search-catch 0.002"
  "E_p8_c5    --press 8  --search-catch 0.005"
  "E_p8_c10   --press 8  --search-catch 0.010"
  "E_p4_c5    --press 4  --search-catch 0.005"
  "E_p4_c10   --press 4  --search-catch 0.010"
)
row=${CELL[$SLURM_ARRAY_TASK_ID]}
echo "### cell: ${row%% *}  [${row#* }]"
python -u peg_insertion_case1.py --errors "${ERRS:-4,6,8}" --episodes "${EPS:-4}" \
    --k-axial-search 800 ${row#* }
