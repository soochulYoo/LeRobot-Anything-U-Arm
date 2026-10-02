#!/bin/bash
# Does variable K_R start to pay exactly where the pad can no longer conform?
#
# Two runs of dome_task disagreed about R=150 and the reason was the PAD, not
# the board: 40 mm (sagitta 5.3 mm > 4 mm of felt give -> cannot lie flat, and
# the oracle won) against 15 mm (0.8 mm -> conforms easily, and a constant won).
# That is either a coincidence of one episode each, or the boundary condition
# this whole line of work has been looking for:
#
#     a constant K_R suffices while the pad can conform;
#     knowing the surface pays once it cannot.
#
# So: both pads, four radii, five seeds, with the glyph actually moved by the
# seed.  The conform/no-conform line falls at R = r^2/(2*give), i.e. 200 mm for
# the 40 mm pad and 28 mm for the 15 mm one -- so the 15 mm pad should conform
# at EVERY radius here and never need the oracle, while the 40 mm pad crosses
# the line between 250 and 150 mm.  That is the prediction being tested.
#SBATCH --job-name=dome_bd
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-39
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dome_bd_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PAD=(0.015 0.040)
RAD=(0.50 0.25 0.15 0.10)
i=$SLURM_ARRAY_TASK_ID
p=$(( i / 20 )); r=$(( (i % 20) / 5 )); s=$(( i % 5 ))
# Set BEFORE python starts: wipe_scene reads it at import and it cannot be an
# argument.  Echoed so the log can never be ambiguous about which pad it used.
export WIPE_PAD_R="${PAD[$p]}"
echo "### pad ${PAD[$p]} m, R ${RAD[$r]} m, seed ${s}"
python -u dome_task.py --radii "${RAD[$r]}" --seed "$s" --jitter-mm 20
