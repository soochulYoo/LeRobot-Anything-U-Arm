#!/bin/bash
# Pick erase_work so the metric does not saturate, and read the mechanism.
#
# At 0.12 the soft constant took 98% of the marks off: no headroom for anything
# to beat it, which is the defect that hid every earlier comparison.  Raise it
# until the best condition sits near 0.7-0.8 and the spread between stiff and
# soft is widest.  The per-band columns come out of the same runs: the design
# predicts the SOFT pad is misaligned on the PLATEAUS (it is still carrying the
# flank's tilt), which is where the cost of softness has to come from.  If
# yld_plat stays small for the soft constant, condition (D) is not biting and
# the plateau/flank lengths need re-tuning, not a longer run.
#SBATCH --job-name=corr_ew
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corr_ew_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_SURFACE=corr WIPE_CORR_PLATEAU=0.030 WIPE_CORR_FLANK=0.100 WIPE_CORR_SLOPE=12
EW=(0.12 0.24 0.40 0.60)
ew="${EW[$SLURM_ARRAY_TASK_ID]}"
echo "### erase_work ${ew}"
python -u corr_gate.py --seed 0 --kr 60,3,0.3 --speed 0.02 --f 4 \
    --erase-work "$ew" --text SSS --letter 0.10
