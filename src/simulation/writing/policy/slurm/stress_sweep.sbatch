#!/bin/bash
#SBATCH --job-name=writing_stress
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.log

# THE STRESS SWEEP: success against how far the paper is from where the policy
# expected it.  One array task per seed.
#
# TILT, NOT dz, and this was measured.  dz is a CONSTANT height offset and the
# policy absorbs it exactly as the operator does -- it descends until it feels
# contact, then presses.  Swept to 20 mm, act_ft_input/seed0 still succeeded and
# in-band moved 99.8 -> 96.6%: a flat curve.  Tilt is deviation DURING contact.
# Over a 37 mm glyph the height ranges w tan(theta), so the sizing rule
# k_n <~ (f - F_lo)/delta caps the stiffness at 1236 / 769 / 509 / 297 N/m at
# 5 / 8 / 12 / 20 deg -- and the demonstrations command 1000 at every tilt.  The
# memorised constant should therefore start losing the band around 8-12 deg.
#
# Why this and not the standard evaluation.  The demonstrations were collected at
# dz = 4 mm and tilt = 5 deg and policy/slurm/eval_policy.sbatch evaluates at the
# same numbers, so there is no distribution shift at all -- which is why every
# structure in policy/runs/COMPARISON.md reproduces the demonstrated stiffness to
# 0.1% and the stiffness axis of that table has no headroom.  A stiffness only
# earns its keep when the surface is NOT where the policy expected it.  Raising
# dz is pure test-time shift: nothing is re-collected and the demos are untouched.
#
# This is the evaluation half of the delta x tilt experiment.  Train the ladder
# with ROBUST_DELTA (policy/slurm/_train.sh) and sweep each of them here; the
# prediction is that a policy trained at delta = 8 mm outlives one trained at
# 2 mm on the right of the curve and is slightly worse at nominal.  If the two
# curves lie on top of each other, --w-robust is not doing anything and that is
# the result.
#
#   NAME=spring_ft_input_d8 sbatch policy/slurm/stress_sweep.sbatch
#   NAME=cross_cond TILT=5,8,12,16,20 sbatch policy/slurm/stress_sweep.sbatch
#   NAME=cross_cond AXIS=dz DZ=4,10,20 sbatch policy/slurm/stress_sweep.sbatch
#
#   TILT      paper tilts in deg (default 5,8,12,16,20; 5 is the demos')
#   AXIS=dz   sweep the height offset instead, with DZ (kept for the record: it
#             is the axis that does NOT move, and showing that is worth one run)
#   DZ        height error in mm, held fixed while tilt sweeps (default 4)
#   EPISODES  per text (default 20)     WORKERS  parallel episodes (default 12)
#
# --time is 8 h because this is EPISODES x texts x |TILT| episodes, five times the
# standard evaluation.  Cut EPISODES before cutting TILT: the shape of the curve
# is the result and a missing point hides the knee.
set -euo pipefail

NAME="${NAME:?set NAME=<run directory under policy/runs>}"
WRITING_DIR="${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WRITING_DIR"
# shellcheck disable=SC1091
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
CKPT="policy/runs/${NAME}/seed${SEED}/model.pkl"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

echo "== $(date)  host $(hostname)  $CKPT"
if [ "${AXIS:-tilt}" = dz ]; then
  SWEEP=(--dz-sweep "${DZ:-4,10,20}" --tilt-deg "${TILT_DEG:-5}")
  echo "== dz ${DZ:-4,10,20} mm at tilt ${TILT_DEG:-5} deg"
else
  SWEEP=(--tilt-sweep "${TILT:-5,8,12,16,20}" --dz "${DZ_FIXED:-0.004}")
  echo "== tilt ${TILT:-5,8,12,16,20} deg at dz ${DZ_FIXED:-0.004} m"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u policy/rollout.py --ckpt "$CKPT" --data "${DATA:-demos/protocol_v1}" \
  --episodes "${EPISODES:-20}" --workers "${WORKERS:-12}" --n-exec "${N_EXEC:-3}" \
  "${SWEEP[@]}" --out "policy/runs/${NAME}/seed${SEED}/stress"
echo "== done $(date)"
