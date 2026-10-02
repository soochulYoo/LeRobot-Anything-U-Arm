#!/bin/bash
# The dome sweep ran ONE episode per cell, and on that single seed R=150 was the
# first environment in the whole rotational-stiffness study where the oracle
# DOMINATED a constant K_R (better force RMSE at equal-or-better erasure).  That
# is too important to rest on one seed.  Five seeds at the two radii where the
# pad cannot conform.
#SBATCH --job-name=dome_rc
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dome_rc_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=(0.15 0.15 0.15 0.15 0.15 0.10 0.10 0.10 0.10 0.10)
S=(0 1 2 3 4 0 1 2 3 4)
r=${R[$SLURM_ARRAY_TASK_ID]}; s=${S[$SLURM_ARRAY_TASK_ID]}
echo "### dome recheck R=$r seed=$s"
python -u dome_task.py --radii "$r" --seed "$s"
