#!/bin/bash
# Shared body of the train_*.sbatch scripts.  Not submitted directly.
#
# Expects CASE (a|b|c|d) and NAME from the calling script.  Everything else is
# overridable from the environment at submission time, e.g.
#   STEPS=50000 sbatch policy/slurm/train_d_cross_cond.sbatch
#
#   WRITING_DIR  src/simulation/writing of the uploaded repo  (default: where you ran sbatch)
#   VENV         python environment with jax[cuda12], flax, optax, h5py
#   DATA         demonstrations, <case>/ep_*.h5 below it       (default: demos/protocol_v1)
#   STEPS        training steps                                (default: 20000)
#   SEED         seed                                          (default: the array index)
#   OUT          checkpoint directory                          (default: policy/runs/<NAME>/seed<SEED>)
#   EXTRA_ARGS   anything else for train.py, e.g. "--no-vision"
set -euo pipefail

: "${CASE:?}" "${NAME:?}"
WRITING_DIR="${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WRITING_DIR"
if [ ! -f policy/train.py ]; then
  echo "policy/train.py not found in $WRITING_DIR -- run sbatch from src/simulation/writing or set WRITING_DIR" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV:-/scratch2/soochul/ManiSkill/.venv}/bin/activate"

DATA="${DATA:-demos/protocol_v1}"
STEPS="${STEPS:-20000}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
OUT="${OUT:-policy/runs/${NAME}/seed${SEED}}"

n_demos=$(find "$DATA" -name 'ep_*.h5' 2>/dev/null | wc -l)
if [ "$n_demos" -eq 0 ]; then
  echo "no demonstrations under $DATA -- copy demos/protocol_v1 to the server first (see policy/README.md)" >&2
  exit 1
fi

echo "== $(date)  host $(hostname)  job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-}"
echo "== case $CASE ($NAME)  seed $SEED  steps $STEPS  data $DATA ($n_demos episodes)  out $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u policy/train.py --case "$CASE" --data "$DATA" --steps "$STEPS" \
  --seed "$SEED" --out "$OUT" ${EXTRA_ARGS:-}
echo "== done $(date)"
