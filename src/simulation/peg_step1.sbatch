#!/bin/bash
# Step 1 of PEG_DEMO_PLAN.md: does scrubbing widen the window the aim leaves?
# One array task per cell: 5 aim errors x {search, aim-and-push}.
#
# dell_cpu, no --gres: this is physics only, nothing renders, so a GPU would sit
# idle and the job would compete for the GPU QoS cap for no reason.
#SBATCH --job-name=peg_s1
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/peg1/%x_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch2/soochul/venvs/writingsim/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
# Keep BLAS single-threaded so the array packs without the tasks fighting for
# cores: each episode is one serial physics rollout.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ERRS=(0 2 4 6 8 0 2 4 6 8)
SEARCH=(4 4 4 4 4 0 0 0 0 0)
e=${ERRS[$SLURM_ARRAY_TASK_ID]}
ss=${SEARCH[$SLURM_ARRAY_TASK_ID]}
echo "### cell: aim error ${e} mm, search_s=${ss}"
python -u peg_insertion_case1.py --errors "$e" --episodes "${EPS:-4}" --search-s "$ss"
