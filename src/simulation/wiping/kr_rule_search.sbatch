#!/bin/bash
# Stage 1: fit rules on one seed, in parallel chunks.
#
# Three boards.  The two 40 mm cells are where the pad CANNOT conform (sagitta
# 5.3 and 8.0 mm against 4 mm of give) -- the regime the boundary hypothesis
# predicted adaptation would pay in, and where it did not.  The 15 mm cell
# conforms easily and is the control: if a searched rule beats the constant
# there too, the win is about the search, not about the surface.
#SBATCH --job-name=kr_rule
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-29
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/kr_rule_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PAD=(0.040 0.040 0.015)
RAD=(0.15  0.10  0.25)
i=$SLURM_ARRAY_TASK_ID
c=$(( i / 10 )); chunk=$(( i % 10 ))
export WIPE_PAD_R="${PAD[$c]}"
echo "### pad ${PAD[$c]} R ${RAD[$c]} chunk ${chunk}"
python -u kr_rule_search.py --radius "${RAD[$c]}" --n 12 --chunk "$((c*10+chunk))"
