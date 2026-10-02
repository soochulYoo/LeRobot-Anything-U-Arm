#!/bin/bash
# One cell, three conditions: does the gate run end to end and does the board
# behave the way gate_feasible.py says it does?
#
# Checks before the full array is worth submitting:
#   the raster is NOT truncated (time_limit raised to 400 s for 0.02 m/s)
#   coverage does not saturate at erase_work 0.12  (if it does, raise it)
#   mis/yld separate between K_R 60 and 0.3        (the board asks something)
#   in_band >= 0.9 everywhere                      (the force loop still holds)
#SBATCH --job-name=corr_chk
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/corr_chk_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The screened cell: pad 15 mm, plateau 30 mm, flank 100 mm, 12 deg, 0.02 m/s,
# 4 N.  t90 3.3 s < flank 5.0 s (the wrist can acquire the tilt) and plateau
# 1.5 s < 2 t90 (it cannot recover before the plateau's marks) -- the window.
export WIPE_SURFACE=corr WIPE_CORR_PLATEAU=0.030 WIPE_CORR_FLANK=0.100 WIPE_CORR_SLOPE=12
echo "### $(python -c "
import sys; sys.path.insert(0,'.')
from patchy_surface import CorrugatedSurface
s=CorrugatedSurface(plateau=0.030,flank=0.100,slope_deg=12.0)
print(f'board pitch {1e3*s.pitch:.0f} mm, height ptp {1e3*(s.height(0.05,0)-s.height(0,0)):.1f} mm')")"
python -u corr_gate.py --seed 0 --kr 60,0.3 --speed 0.02 --f 4 --erase-work 0.12 \
    --text SSS --letter 0.10
