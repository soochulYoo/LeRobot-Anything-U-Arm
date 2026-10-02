#!/bin/bash
#SBATCH --job-name=peg_bc
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pegbc/%x_%A_%a.log

# Per-episode best constant stiffness vs the best global constant on peg
# insertion.  Rationale in peg_best_constant.py; in short, PEG_DEMO_PLAN.md
# section 10 compared ONE schedule against a ladder whose softest rung won, so
# the optimum may lie below the ladder and a single losing schedule says nothing
# about schedules in general.  This has no rule in it and the ladder goes to 100.
#
# ONE (stiffness, clearance) CELL PER TASK = 32 episodes, which is
# peg_step4_clearance.sbatch's proven footprint.  The first attempt ran all 160
# episodes of a clearance in one process and SEGFAULTED at ~60 -- `episode()`
# builds and closes a SAPIEN env every call and the process does not survive that
# many.  There was no Python traceback, so a traceback-only watch missed it.
#
#   10 tasks = 5 stiffnesses x 2 clearances.  Then:
#     python3 peg_best_constant.py --aggregate "peg_bc_k*_cl*.json"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source "/scratch2/soochul/venvs/writingsim/bin/activate"

LADDER=(100 200 400 800 1600)
i=${SLURM_ARRAY_TASK_ID:-0}
K=${LADDER[$((i % 5))]}
CL=$([ $((i / 5)) -eq 0 ] && echo 2 || echo 1)

echo "== $(date) host $(hostname)  k_axial ${K} N/m at ${CL} mm clearance"
python -u peg_best_constant.py --ladder "$K" --clearances "$CL" \
  --errors "${ERRORS:-0,2,4,6}" --seeds "${SEEDS:-8}" \
  --out "peg_bc_k${K}_cl${CL}.json"
echo "== done $(date)"
