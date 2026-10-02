#!/bin/bash
# Why does R=150 give a different answer alone than it does inside the sweep?
#
# The sweep sets the radius through an environment variable (WIPE_DOME_R) that
# the scene reads when it is built, and runs all four radii in one process.  If
# anything survives between radii -- a cached mesh, a module-level surface, a
# controller state -- then the rows after the first describe a board that is not
# the one in their label.  This runs R=0.15 alone and after two different
# predecessors; if the three disagree, the sweep's rows cannot be trusted.
#SBATCH --job-name=dome_dg
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dome_dg_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=("0.15" "0.50,0.15" "0.25,0.15")
echo "### radii: ${R[$SLURM_ARRAY_TASK_ID]}  (compare the R=150 block)"
python -u dome_task.py --radii "${R[$SLURM_ARRAY_TASK_ID]}"
