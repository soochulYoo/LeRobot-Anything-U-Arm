#!/bin/bash
#SBATCH --job-name=writing_verify
#SBATCH --partition=suma_a6000,base_suma_rtx3090,gigabyte_a6000,dell_rtx3090
#SBATCH --exclude=cs-gpu-01,node29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

# Checks the evaluation environment before a sweep leans on it: the simulator's
# own 30 tests under this venv's numpy 2, then a two-episode rollout of a real
# checkpoint.  Run it after rebuilding /scratch2/soochul/venvs/writingsim.
#   CKPT  checkpoint for the rollout smoke test (default: ft_input seed0)
set -euo pipefail
cd "${WRITING_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${VENV:-/scratch2/soochul/venvs/writingsim}/bin/activate"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json}"

echo "== versions"
python -c "
import numpy, scipy, torch, sapien, mani_skill, jax, flax
print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| torch', torch.__version__)
print('sapien', sapien.__version__, '| mani_skill', mani_skill.__version__)
print('jax', jax.__version__, '| flax', flax.__version__, '| jax devices', jax.devices())
"
echo "== simulator tests"
python -u tests.py

CKPT="${CKPT:-policy/runs/ft_input/seed0/model.pkl}"
if [ -f "$CKPT" ]; then
  echo "== rollout smoke test on $CKPT (2 episodes, 1 text)"
  python -u policy/rollout.py --ckpt "$CKPT" --texts S --episodes 2 --workers 2 \
    --out "$(dirname "$CKPT")/eval_smoke"
else
  echo "== no checkpoint at $CKPT yet, skipping the rollout smoke test" >&2
fi
echo "== done $(date)"
