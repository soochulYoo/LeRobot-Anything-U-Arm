#!/bin/bash
#SBATCH --job-name=writing_surprise
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log

# Force as surprise: the cross-fitted expectation model (policy/surprise.py).
#
# A PREREQUISITE, not a variant: it writes
#   <DATA>/_surprise_<IMG>_k<FOLDS>.npz
# which the training jobs then read through SURPRISE=.  One job for the whole
# sweep -- the folds are over episodes, not over seeds, so every policy seed
# uses the same out-of-fold residuals.
#
#   mkdir -p logs && sbatch policy/slurm/surprise.sbatch
#   FOLDS=8 STEPS=8000 sbatch policy/slurm/surprise.sbatch
#
# Read the R^2 it prints before building on it.  On demos/protocol_v1, 4 folds,
# 2500 steps:  normal axis 0.87, tangential axes -0.20 and 0.00.  That split is
# the result -- the camera predicts the pressing force from geometry and knows
# NOTHING about friction, because mu is randomised 0.2-0.5 and invisible.  So the
# tangential residual is pure "feel, not look", which is what the feature is for.
# An R^2 near 1.0 on every axis would mean the residual is noise and the extra
# channels are not worth their cost.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f policy/surprise.py ]; then
  echo "run sbatch from src/simulation/writing" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV:-/scratch2/soochul/venvs/jaxgpu}/bin/activate"
LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '/opt/ohpc/pub/apps/cuda' | paste -sd: -)"
export LD_LIBRARY_PATH
echo "== $(date)  host $(hostname)  job ${SLURM_JOB_ID:-local}"
# DEPLOY_ONLY=1 fits only the all-episode expectation model used at INFERENCE
# and leaves the npz alone -- the right mode once policies have been trained
# against that npz, since regenerating it would change their targets.
python -u policy/surprise.py --data "${DATA:-demos/protocol_v1}" \
  --folds "${FOLDS:-4}" --steps "${STEPS:-4000}" --batch "${BATCH:-64}" \
  ${DEPLOY_ONLY:+--deploy-only}
echo "== done $(date)"
