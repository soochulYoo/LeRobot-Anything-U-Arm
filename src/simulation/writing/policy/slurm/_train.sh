#!/bin/bash
# Shared body of the train_*.sbatch scripts.  Not submitted directly.
#
# Expects CASE (a|b|c|d) and NAME from the calling script.  Everything else is
# overridable from the environment at submission time, e.g.
#   STEPS=50000 sbatch policy/slurm/train_d_cross_cond.sbatch
#
#   WRITING_DIR  src/simulation/writing of the uploaded repo  (default: where you ran sbatch)
#   VENV         python environment with jax[cuda12], flax, optax, h5py
#                (default /scratch2/soochul/venvs/jaxgpu -- kept separate from the
#                 ManiSkill venv, whose numpy/scipy a jax install would upgrade)
#   ARCH         flow (CoFA fields) or act (Comp-ACT)          (default: flow)
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
source "${VENV:-/scratch2/soochul/venvs/jaxgpu}/bin/activate"

# The cluster's cuda/12.8 module puts /opt/ohpc/.../cuda/12.8/lib64 ahead of the
# CUDA libraries pip installed next to jaxlib.  jaxlib then fails to load
# cuSPARSE, prints one warning and runs on the CPU at a tenth of the speed.
# Drop the system CUDA entries and keep the driver ones (/usr/lib/nvidia).
LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '/opt/ohpc/pub/apps/cuda' | paste -sd: -)"
export LD_LIBRARY_PATH

ARCH="${ARCH:-flow}"
DATA="${DATA:-demos/protocol_v1}"
STEPS="${STEPS:-20000}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"

# The three axes that are ORTHOGONAL to --case, so the same four sbatch files
# serve every combination rather than there being 4 x 2 x 2 of them:
#   LAYOUT    legacy (x_d + log K) | spring (x_ref + f_d + log K)
#   W_ROBUST  weight of the stiffness-sizing term; spring only, 0 = K unsupervised
#   SURPRISE  a _surprise_*.npz from policy/surprise.py -> ft_hist gains the
#             expected wrench and the z-score (6 extra channels)
# NAME is extended to match model.run_name(), so no two combinations can land in
# one checkpoint directory.
LAYOUT="${LAYOUT:-legacy}"
EXTRA="--layout ${LAYOUT}"
if [ "$LAYOUT" = spring_rel ]; then
  NAME="springrel_${NAME}"
fi
if [ "$LAYOUT" = spring ] || [ "$LAYOUT" = spring_rel ]; then
  [ "$LAYOUT" = spring ] && NAME="spring_${NAME}"
  EXTRA="$EXTRA --w-robust ${W_ROBUST:-0}"
  if [ "${W_ROBUST:-0}" != 0 ]; then
    # must match model.run_name(): delta is the swept hypothesis, so it is in
    # the directory name.  Default 0.002 m mirrors train.py's.
    RD="${ROBUST_DELTA:-0.002}"
    EXTRA="$EXTRA --robust-delta $RD"
    NAME="${NAME}_d$(python3 -c "print(round(1000*$RD))")"
  fi
fi
# The control for the eight-variant ranking, and (d)'s sampling schedule.
# NAME must track model.run_name() exactly or eval reads the wrong directory.
if [ -n "${D_LEAD:-}" ] && [ "${D_LEAD}" != 0 ]; then
  EXTRA="$EXTRA --d-lead ${D_LEAD}"
  NAME="${NAME}_lead${D_LEAD}"
fi
if [ -n "${NO_FORCE:-}" ]; then
  EXTRA="$EXTRA --no-force"
  NAME="${NAME}_nf"
  if [ -n "${NO_FORCE_TARGET:-}" ]; then EXTRA="$EXTRA --no-force-target"; NAME="${NAME}t"; fi
elif [ -n "${NO_FORCE_TARGET:-}" ]; then
  EXTRA="$EXTRA --no-force-target"
  NAME="${NAME}_nft"
fi
if [ -n "${SURPRISE:-}" ]; then
  if [ ! -f "$SURPRISE" ]; then
    echo "SURPRISE=$SURPRISE does not exist -- run policy/slurm/surprise.sbatch first" >&2
    exit 1
  fi
  NAME="${NAME}_sur"
  EXTRA="$EXTRA --surprise ${SURPRISE}"
fi
OUT="${OUT:-policy/runs/${NAME}/seed${SEED}}"

n_demos=$(find "$DATA" -name 'ep_*.h5' 2>/dev/null | wc -l)
if [ "$n_demos" -eq 0 ]; then
  echo "no demonstrations under $DATA -- copy demos/protocol_v1 to the server first (see policy/README.md)" >&2
  exit 1
fi

# A few nodes advertise a GPU the driver cannot open.  Training would silently
# fall back to CPU and take ten times as long, so fail fast and let Slurm
# requeue the task somewhere healthy.
if [ -n "${SLURM_JOB_ID:-}" ] && ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
  echo "no usable GPU for jax on $(hostname), requeueing" >&2
  scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
  exit 1
fi

echo "== $(date)  host $(hostname)  job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-}"
echo "== case $CASE ($NAME)  arch $ARCH  seed $SEED  steps $STEPS  data $DATA ($n_demos episodes)  out $OUT"
echo "== layout/robust/surprise: $EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
python -u policy/train.py --case "$CASE" --arch "$ARCH" --data "$DATA" --steps "$STEPS" \
  --seed "$SEED" --out "$OUT" ${EXTRA} ${EXTRA_ARGS:-}
echo "== done $(date)"
