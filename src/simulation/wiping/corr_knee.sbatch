#!/bin/bash
# The design works; the METRIC has no resolution.  Two findings from corr_ew:
#
#   The schedule does what it was built to do.  Per band, the oracle sits near
#   the soft constant on the flanks (mis 4.6 vs 3.4 deg) and near the stiff one
#   on the plateaus (1.5 vs 0.6), which no constant can do -- stiff is 11.8 on
#   the flanks, soft is 3.4 on the plateaus.
#   But erase_work is a CLIFF, not a knob: 0.12 gives 78-98% and 0.24 gives
#   3-4%.  Each mark is passed a fixed number of times, so the wear it collects
#   is concentrated and coverage is a step function of the threshold.  At 0.12
#   the soft constant is already at 98%, leaving the oracle 2 pp to win by.
#
# So: find the knee of the cliff, where coverage has the most dynamic range, and
# at the same time give the PLATEAUS more of the path (50/80 against 30/100, so
# the plateau is half the traverse instead of a third) since that is where the
# soft constant pays and the oracle does not.  Both still satisfy the window:
# 50/80 has t_flank 4.0 s > t90 3.3 s and t_plat 2.5 s < 2 t90 = 6.7 s.
#SBATCH --job-name=corr_knee
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corr_knee_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PLAT=(0.030 0.050); FLNK=(0.100 0.080)
EW=(0.14 0.16 0.18)
i=$SLURM_ARRAY_TASK_ID
g=$(( i / 3 )); e=$(( i % 3 ))
export WIPE_SURFACE=corr WIPE_CORR_SLOPE=12
export WIPE_CORR_PLATEAU="${PLAT[$g]}" WIPE_CORR_FLANK="${FLNK[$g]}"
echo "### plateau ${PLAT[$g]} flank ${FLNK[$g]} erase_work ${EW[$e]}"
python -u corr_gate.py --seed 0 --kr 60,3,0.3 --speed 0.02 --f 4 \
    --erase-work "${EW[$e]}" --text SSS --letter 0.10
