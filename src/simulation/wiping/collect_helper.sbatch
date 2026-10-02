#!/bin/bash
# The closed-loop comparison: v0 and v1 driving the stiffness, on the SAME seeds as the
# bad / normal / good tiers already collected.
#
# Five arms in one axis, differing in exactly one thing -- who chooses the level:
#
#   scripted bad     the protocol's levels, 0.70 s late      erased 0.713  success 0.38
#   scripted normal                         0.35 s late      erased 0.896  success 0.73
#   scripted good                           on time          erased 0.967  success 0.88
#   helper v0        a model trained to IMITATE the bad operator's applied K
#   helper v1        a model trained on the hindsight labeler's K, from the SAME bad logs
#
# v1 - bad against good - bad is the number this whole branch exists to produce.  The ramp,
# the energy tank, the compliance score and the recorder are the same path in all five, and
# the helper owns the rotational axis alone (--helper-axes r): see helper_user.py on why
# handing it the other two would measure the labeler's weakest rule instead of its
# strongest.
#SBATCH --job-name=wipe_helper
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wipe_helper_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_PAD_R=0.040
export STIFFNESS_HELPER=/scratch2/soochul/TeleopHelper

V=(v0 v1)
T=${V[$SLURM_ARRAY_TASK_ID]}
CKPT="$STIFFNESS_HELPER/runs/w40_$T/model.pt"
test -f "$CKPT" || { echo "no checkpoint at $CKPT"; exit 1; }
echo "### helper $T   $CKPT   rotational axis only, curved board, 40 mm pad"
python -u protocol.py --board curved --texts S 7 "<star>" \
    --helper "$CKPT" --helper-version "$T" --helper-axes r \
    --per-case 20 --attempts 20 \
    --keep-failed --min-compliance 0 --headless --workers 12 \
    --video 2 --seed-base 70000 \
    --out "demos/helper/$T"
