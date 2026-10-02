#!/bin/bash
# The three tiers on the curved board with a 40 mm pad: 60 episodes each.
#
# Mirrors the writing set (demos/scripted/{bad,normal,good}, 3 texts x 20 attempts) so
# the same loader reads both.  --keep-failed --min-compliance 0 is deliberate: a bad
# tier's failures ARE the dataset, and --attempts is what lets a tier that mostly fails
# terminate (--per-case counts kept demos).
#
# The pad is 40 mm because that is the only regime in this repository where a stiffness
# schedule was measured to beat the best constant: `ROTATIONAL_STIFFNESS.md` has
# best-constant 72.6% against oracle 83.9% erased, +11.3 pp, and quantifies the existence
# condition -- a 15 mm pad sees 5.3 deg of normal swing and nothing happens, a 40 mm pad
# sees 13.1 deg and everything does.  A flat board or a small pad collects data about a
# task that does not need the thing being studied.
#SBATCH --job-name=wipe_tiers
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/wipe_tiers_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_PAD_R=0.040

TIERS=(bad normal good)
T=${TIERS[$SLURM_ARRAY_TASK_ID]}
echo "### tier $T   curved board, 40 mm pad, 3 texts x 20 attempts"
python -u protocol.py --board curved --texts S 7 "<star>" \
    --scripted "$T" --per-case 20 --attempts 20 \
    --keep-failed --min-compliance 0 --headless --workers 12 \
    --video 2 --seed-base 70000 \
    --out "demos/scripted/$T"
