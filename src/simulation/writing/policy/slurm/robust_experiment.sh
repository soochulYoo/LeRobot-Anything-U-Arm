#!/bin/bash
# THE delta x tilt EXPERIMENT: the one falsifiable test of --w-robust that runs on
# the demonstrations we already have.
#
# The problem it works around.  In policy/runs/COMPARISON.md every structure
# commands the demonstrated stiffness to 0.1% and `acc xy`/`acc z` sit at 98%,
# because protocol.EXPECTED makes K a four-entry function of the writer's phase
# and a phase classifier reproduces it exactly.  Nothing about stiffness can be
# measured on that dataset -- so stop asking the policy to reproduce a K and ask
# instead what its K is FOR.
#
# The hypothesis.  --robust-delta is a requirement: "keep the force in the band
# when the surface is up to delta from where the policy expected it".  So
#
#     a policy trained at delta = 8 mm should outlive one trained at delta = 2 mm
#     when evaluated at tilt = 12-20 deg, and should be slightly WORSE at 5 deg,
#
# because it gives up force precision at nominal to buy tolerance.  Both curves
# lying on top of each other falsifies the term; the 8 mm arm losing everywhere
# falsifies it the other way.  Four arms, so the layout and the term are
# separable:
#
#   legacy              the labels the first eight runs used        (control)
#   spring w_robust 0   the layout alone, K unsupervised            (control)
#   spring w_robust 1 delta 2mm    K sized for the demos' own range
#   spring w_robust 1 delta 8mm    K sized for twice beyond it
#
# Run from anywhere in the repo:  bash policy/slurm/robust_experiment.sh
#   ARCH=act|flow   family (default act: it is at ceiling at nominal dz, so the
#                   only thing that can move on the curve is robustness)
#   CASE=a          structure (default a, the simplest and the strongest)
#   ARRAY=0-2       seeds
#   TILT=...        sweep points in deg (default the sbatch's 5,8,12,16,20).  Tilt
#                   and not dz: dz is a constant offset the contact search absorbs
#                   (measured flat to 20 mm), tilt is deviation DURING contact and
#                   the demos' k_n = 1000 N/m is already at its limit at 5 deg.
#   NO_SWEEP=1      train only
#   STEPS=20000     training steps
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> src/simulation/writing
mkdir -p logs

ARCH="${ARCH:-act}"
CASE="${CASE:-a}"
declare -A CASE_NAME=([a]=ft_input [b]=uni_dir [c]=unified [d]=cross_cond)
base="${CASE_NAME[$CASE]:?CASE must be a, b, c or d}"
[ "$ARCH" = act ] && base="act_${base}"
script="policy/slurm/train_$([ "$ARCH" = act ] && echo act_)${CASE}_${CASE_NAME[$CASE]}.sbatch"
[ -f "$script" ] || { echo "no sbatch at $script" >&2; exit 1; }

# arm := "LAYOUT:W_ROBUST:ROBUST_DELTA", and the run name each one lands in
arms=("legacy:0:" "spring:0:" "spring:1:0.002" "spring:1:0.008")

echo "structure $CASE ($base), arch $ARCH, seeds ${ARRAY:-0-2}"
for arm in "${arms[@]}"; do
  IFS=: read -r layout wrob rd <<<"$arm"
  name="$base"
  if [ "$layout" = spring ]; then
    name="spring_${name}"
    if [ "$wrob" != 0 ]; then name="${name}_d$(python3 -c "print(round(1000*$rd))")"; fi
  fi
  # the legacy arm is already trained; do not burn a GPU re-running it
  if [ "$layout" = legacy ] && [ -f "policy/runs/${name}/seed0/model.pkl" ]; then
    printf '  %-28s train SKIPPED (already in policy/runs/%s)\n' "$name" "$name"
    tid=""
  else
    tid=$(LAYOUT="$layout" W_ROBUST="$wrob" ROBUST_DELTA="$rd" ARCH="$ARCH" \
          sbatch --parsable ${ARRAY:+--array="$ARRAY"} "$script")
    printf '  %-28s train %s\n' "$name" "$tid"
  fi
  if [ -n "${NO_SWEEP:-}" ]; then continue; fi
  sid=$(NAME="$name" ${TILT:+TILT="$TILT"} sbatch --parsable ${ARRAY:+--array="$ARRAY"} \
        ${tid:+--dependency=aftercorr:"$tid"} policy/slurm/stress_sweep.sbatch)
  printf '  %-28s sweep %s\n' "" "$sid"
done

cat <<'MSG'

Each sweep writes policy/runs/<name>/seed<k>/stress/tilt_sweep.json.
Collect the four curves with:
    python3 policy/plot_stress.py --runs policy/runs
MSG
