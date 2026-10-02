#!/bin/bash
#SBATCH --job-name=retime_demos
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --array=0-24
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log

# S4 step 3, part 1: the rate-augmented demonstration set.
#
# Every demonstration is replayed through the real controller on its own paper
# at several clock rates, with the stiffness left EXACTLY as demonstrated, and
# the episodes that still pass the simulator's ink, force-band and tear criteria
# are written as new demonstrations.  The contact force in them is measured, not
# relabeled -- which is the whole reason not to use Theorem 1's c^2 K relabeling
# here, since S3 showed it tears the paper and S4 showed it also corrupts the
# policy's own observation of contact.
#
# No GPU: the cameras go through SAPIEN's CPU rasterizer, dell_cpu is idle, and
# the account's GPU QoS is already saturated by the training array.
#
#   DATA   source demonstrations   (default demos/protocol_v1)
#   OUT    destination            (default demos/protocol_v1_retimed)
#   RATES  clock rates            (default "1 1.5 2 2.5 3 4")
set -euo pipefail

WRITING_DIR="${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WRITING_DIR"
# shellcheck disable=SC1091
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs

N="${SLURM_ARRAY_TASK_COUNT:-25}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "== $(date)  host $(hostname)  shard $I/$N"

python -u retime_demos.py \
  --data "${DATA:-demos/protocol_v1}" \
  --out "${OUT:-demos/protocol_v1_retimed}" \
  --c ${RATES:-1 1.5 2 2.5 3 4} \
  --shard "$I" --shards "$N"

echo "== done $(date)"
