#!/bin/bash
#SBATCH --job-name=writing_eval
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log

# Closed-loop evaluation of one trained structure, one array task per seed:
# 20 unseen paper randomizations per text, in the simulator the demos came from.
# The GPU is for SAPIEN's cameras; the policy itself runs on CPU JAX in each worker.
# Submit from src/simulation/writing, naming the structure:
#   NAME=cross_cond sbatch policy/slurm/eval_policy.sbatch
# Writes policy/runs/<NAME>/seed<k>/eval/{results.json, rollouts.png}.
#
#   EPISODES  per text (default 20)     WORKERS  parallel episodes (default 12)
#   N_EXEC    set-points executed per sampled chunk (default 3 of 10)
set -euo pipefail

NAME="${NAME:?set NAME=ft_input|uni_dir|unified|cross_cond}"
WRITING_DIR="${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WRITING_DIR"
# shellcheck disable=SC1091
# The simulator and jax in one environment: jax needs numpy >= 2, which the
# ManiSkill venv pins below.  VENV_SIM is that environment (see policy/README.md).
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
CKPT="policy/runs/${NAME}/seed${SEED}/model.pkl"
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

echo "== $(date)  host $(hostname)  $CKPT"
# D_LEAD re-evaluates THESE weights with (d)'s wrench field N steps ahead, and
# OUT_SUB puts the result in a sibling of eval/ so the two schedules can be
# compared on one checkpoint instead of across two trainings.
EXTRA=()
[ -n "${D_LEAD:-}" ] && EXTRA+=(--d-lead "${D_LEAD}")
[ -n "${OUT_SUB:-}" ] && EXTRA+=(--out "policy/runs/${NAME}/seed${SEED}/${OUT_SUB}")
python -u policy/rollout.py --ckpt "$CKPT" --data "${DATA:-demos/protocol_v1}" \
  --episodes "${EPISODES:-20}" --workers "${WORKERS:-12}" --n-exec "${N_EXEC:-3}" \
  "${EXTRA[@]}"
echo "== done $(date)"
