#!/bin/bash
#SBATCH --job-name=speedup_diag
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log

# S4 step 2: WHICH observation does retiming break?
#
# At rate c the policy sees its own twist c times larger, its own force history
# over 1/c the physical window, and -- under a gain rule -- its own stiffness
# c^2 larger.  Each ablation hands exactly one of those back at its 1x value and
# asks whether the behaviour returns.  None of them is a controller; they are a
# diagnosis, and they decide whether the retrained policy needs to be TOLD the
# rate (one input explains it) or retrained on retimed data (they all do).
#
# This wants the cameras (obs_mode="rgb"), which normally argues for a GPU --
# but the GPU QoS is saturated by this account's own training array, and SAPIEN
# falls back to its CPU rasterizer, which is what every rollout in this study has
# actually used.  dell_cpu is idle, so the job starts now instead of queueing
# behind the training it would otherwise slow down.
#
#   CKPT     checkpoint under test   (default act_ft_input/seed0)
#   WORKERS  parallel episodes       (default 12)
set -euo pipefail

WRITING_DIR="${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WRITING_DIR"
# shellcheck disable=SC1091
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/speedup

CKPT="${CKPT:-policy/runs/act_ft_input/seed0/model.pkl}"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
echo "== $(date)  host $(hostname)  $CKPT"

# `fixed` asks what breaks the plan the policy generates; `uniform` asks whether
# the stiffness OBSERVATION is what kills it at 1.25x, which is the one
# hypothesis S4 can already name.
python -u speedup_contact.py --policy "$CKPT" \
  --attempts 500 501 --c 1 1.25 1.5 2 2.5 \
  --rules fixed uniform --obs-ablate none vel k ft all \
  --workers "${WORKERS:-12}" \
  > logs/speedup/S4_diag.txt 2>&1 || true
tail -40 logs/speedup/S4_diag.txt
echo "== done $(date)"
