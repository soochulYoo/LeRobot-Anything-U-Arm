#!/bin/bash
#SBATCH --job-name=writing_compare
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.log

# The last link of submit_all.sh's chain: read every run's train_log.json and
# eval/results.json, write the comparison table, and draw every evaluation
# episode on one page.  No GPU and no simulator; the venv is only for numpy and
# matplotlib.
#   RUNS  checkpoint root (default policy/runs)
#   OUT   markdown table  (default policy/runs/COMPARISON.md)
set -euo pipefail
cd "${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUNS="${RUNS:-policy/runs}"
python3 -u policy/compare.py --runs "$RUNS" --out "${OUT:-$RUNS/COMPARISON.md}"
"${VENV:-/scratch2/soochul/venvs/writingsim}/bin/python" -u policy/plot_overlay.py \
  --runs "$RUNS" --out "$RUNS/OVERLAY.png"
