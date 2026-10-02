#!/bin/bash
# Set the scripted tiers' two delays by measuring, the way writing/scripted.py did.
#
# One delay is swept with the other held at its "good" value, so the table says what each
# one costs on its own.  The expectation, carried over from writing: the seeing delay
# decides how well the glyph comes off, and the level reaction does not touch it at all,
# because the operator's haptic loop presses against what it MEASURES and the force it
# settles on is its own choice whatever K is.  On the CURVED board with a 40 mm pad that
# expectation may break on one axis -- K_R is the one `ROTATIONAL_STIFFNESS.md` measured
# a task effect for (14.5% -> 93.5% erased with the damping corrected) -- and a late K_R
# costing erasure where a late k_z cannot would itself be the result.
#SBATCH --job-name=tier_sweep
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tier_sweep_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WIPE_PAD_R=0.040          # the 40 mm pad: the regime K_R matters in at all

MOTION=(0.10 0.20 0.30 0.45 0.60 0.80)
STIFF=(0.0 0.2 0.5 1.0 2.0 4.0)
i=${SLURM_ARRAY_TASK_ID}
if [ "$i" -lt 6 ]; then
  M=${MOTION[$i]}; S=0.2; TAG="motion_$M"      # sweep seeing, hold the levels good
else
  M=0.10; S=${STIFF[$((i-6))]}; TAG="stiff_$S" # sweep the levels, hold seeing good
fi
echo "### $TAG   seeing ${M}s   level reaction ${S}s   curved board, 40 mm pad"
python -u protocol.py --board curved --texts S 7 "<star>" \
    --scripted "$M" "$S" --per-case 5 --attempts 5 \
    --keep-failed --min-compliance 0 --headless --workers 4 \
    --no-cameras --seed-base 90000 \
    --out "demos/tier_sweep/$TAG"
