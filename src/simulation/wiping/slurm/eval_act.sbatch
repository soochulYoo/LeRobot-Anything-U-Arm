#!/bin/bash
#SBATCH --job-name=wipe_eval
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.log
#
# Closed-loop evaluation of one variant's seeds on unseen boards.
#
# The GPU is for SAPIEN's cameras, not for the policy: inference is on the CPU
# (rollout.py pins JAX to it) because a dozen worker processes would otherwise
# each reserve most of the GPU the renderer needs.  The venv is the ManiSkill
# one, which carries a CPU JAX for exactly this.
#
#   NAME      run directory under wiping/runs/
#   EPISODES  per text (3 texts), default 10
#   BOARD     curved | flat          DEMOS  the demonstrations to compare against
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
NAME="${NAME:?set NAME}"
for ck in runs/"$NAME"/seed*/model.pkl; do
  seed=$(basename "$(dirname "$ck")")
  echo "== $(date) $NAME $seed"
  python -u rollout.py --ckpt "$ck" --board "${BOARD:-curved}" \
      --episodes "${EPISODES:-10}" --workers "${WORKERS:-10}" \
      --demos "${DEMOS:-demos/wipe_curved}" \
      --out "runs/$NAME/$seed/eval/results.json"
done
echo "== done $(date)"
