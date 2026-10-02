#!/bin/bash
# The gate.  One job per seed; six constants and the oracle in each.
#
# The best constant is selected on the SAME seeds it is scored on, which is a
# bias in the BASELINE's favour -- deliberately, since the hypothesis under test
# is that the oracle beats it.  The oracle has nothing fitted, so it needs no
# held-out split; the rule search (kr_rule_search.py) is where one is required.
#SBATCH --job-name=corr_gate
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-7
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/corr_gate_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_SURFACE=corr WIPE_CORR_PLATEAU=0.030 WIPE_CORR_FLANK=0.100 WIPE_CORR_SLOPE=12
# erase_work 0.16 is the one cell of twelve where the oracle came out ahead on a
# single seed (+5.4 pp).  Every other cell had it level or behind, and the
# hard-pad sweep had it behind in all six, so this run is here to settle whether
# that +5.4 was the seed or the schedule.  Eight seeds, and the verdict needs the
# gap to clear both 5 pp and twice the seed spread.
echo "### seed ${SLURM_ARRAY_TASK_ID}, corrugation 30/100 mm at 12 deg, 0.02 m/s, 4 N, ew 0.16"
python -u corr_gate.py --seed "$SLURM_ARRAY_TASK_ID" --speed 0.02 --f 4 \
    --erase-work "${EW:-0.16}" --text SSS --letter 0.10
