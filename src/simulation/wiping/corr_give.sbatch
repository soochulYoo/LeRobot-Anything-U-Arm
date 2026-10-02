#!/bin/bash
# The last structural knob: how hard the felt is.
#
# The oracle's per-band advantage is real and large (plateau misalignment 1.5
# against the soft constant's 3.4 deg) and it converts into NO coverage, across
# six (geometry, erase_work) cells.  The reason is a threshold in the erasure
# model itself: a tilt whose height change across the pad is under `pad_give` is
# squashed out and costs nothing, and
#
#     atan(pad_give / 2 r) = atan(4 mm / 30 mm) = 7.6 deg
#
# is larger than the soft wrist's entire error.  Coverage is blind to everything
# the schedule is fixing.  A harder pad -- a squeegee or a scraper rather than a
# sponge -- lowers the threshold under the error: 1 mm gives 1.9 deg.
#
# A harder pad also concentrates the force into a smaller patch, so the wear per
# mark goes up and the erase_work cliff moves; hence the second axis.
#SBATCH --job-name=corr_give
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corr_give_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_SURFACE=corr WIPE_CORR_SLOPE=12 WIPE_CORR_PLATEAU=0.030 WIPE_CORR_FLANK=0.100
GIVE=(0.001 0.002); EW=(0.06 0.10 0.14)
i=$SLURM_ARRAY_TASK_ID
g=$(( i / 3 )); e=$(( i % 3 ))
echo "### pad_give ${GIVE[$g]} erase_work ${EW[$e]}"
python -u corr_gate.py --seed 0 --kr 60,3,0.3 --speed 0.02 --f 4 \
    --pad-give "${GIVE[$g]}" --erase-work "${EW[$e]}" --text SSS --letter 0.10
