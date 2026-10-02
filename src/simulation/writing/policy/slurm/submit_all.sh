#!/bin/bash
# Submit the whole comparison as one dependency chain:
#
#   train_<s>.sbatch (array 0-2)  --aftercorr-->  eval_policy.sbatch (array 0-2)
#                                                        |
#                                       all four --afterany--> compare.sbatch
#
# aftercorr pairs the two arrays task by task, so seed k is evaluated as soon as
# seed k has finished training, and a seed that fails does not hold up the rest.
# The final compare job uses afterany, so a partial sweep still gets a table.
#
# Run from anywhere inside the repo:  bash policy/slurm/submit_all.sh
# Overrides pass straight through, e.g.  STEPS=50000 bash policy/slurm/submit_all.sh
#   FAMILY=flow|act|both  which family to submit (default both, 24 training jobs)
#   ARRAY=3-7   seeds to run, as a Slurm array range (default 0-2, the script's own)
#   NO_EVAL=1   training only
#   EPISODES=..  WORKERS=..  N_EXEC=..   forwarded to the evaluation
#
# The action layout and the two input/loss options, forwarded to _train.sh:
#   LAYOUT=legacy|spring   default legacy, the labels the first runs used
#   W_ROBUST=1.0           the stiffness-sizing term; spring only.  Under spring,
#                          K gets NO gradient from imitation (x = x_ref makes the
#                          spring term identically zero on every sample), so
#                          W_ROBUST=0 and W_ROBUST=1 is the ablation that says
#                          whether K was ever being learned or only absorbing
#                          gradient meant for the force.
#   SURPRISE=<npz>         from policy/slurm/surprise.sbatch; adds the expected
#                          wrench and the z-score to ft_hist
# The three combinations worth running against the existing eight:
#   LAYOUT=spring W_ROBUST=0   bash policy/slurm/submit_all.sh   # layout alone
#   LAYOUT=spring W_ROBUST=1   bash policy/slurm/submit_all.sh   # + K a reason
#   LAYOUT=spring W_ROBUST=1 SURPRISE=demos/protocol_v1/_surprise_64_k4.npz \
#                              bash policy/slurm/submit_all.sh   # + surprise
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> src/simulation/writing, the SLURM_SUBMIT_DIR
mkdir -p logs

declare -A NAME=([a_ft_input]=ft_input [b_uni_dir]=uni_dir [c_unified]=unified [d_cross_cond]=cross_cond)

SCRIPTS=()
case "${FAMILY:-both}" in
  flow) for s in a_ft_input b_uni_dir c_unified d_cross_cond; do SCRIPTS+=("$s:"); done ;;
  act)  for s in a_ft_input b_uni_dir c_unified d_cross_cond; do SCRIPTS+=("$s:act_"); done ;;
  both) for s in a_ft_input b_uni_dir c_unified d_cross_cond; do SCRIPTS+=("$s:" "$s:act_"); done ;;
  *) echo "FAMILY must be flow, act or both" >&2; exit 2 ;;
esac

eval_ids=()
for entry in "${SCRIPTS[@]}"; do
  s=${entry%%:*}; pre=${entry##*:}
  script="policy/slurm/train_${pre:+act_}${s}.sbatch"
  tid=$(sbatch --parsable ${ARRAY:+--array="$ARRAY"} "$script")
  name="${pre}${NAME[$s]}"
  # must match _train.sh and model.run_name(), or eval reads the wrong directory
  if [ "${LAYOUT:-legacy}" = spring ]; then name="spring_${name}"; fi
  if [ "${LAYOUT:-legacy}" = spring_rel ]; then name="springrel_${name}"; fi
  if { [ "${LAYOUT:-legacy}" = spring ] || [ "${LAYOUT:-legacy}" = spring_rel ]; } \
     && [ "${W_ROBUST:-0}" != 0 ]; then
    name="${name}_d$(python3 -c "print(round(1000*${ROBUST_DELTA:-0.002}))")"
  fi
  if [ -n "${D_LEAD:-}" ] && [ "${D_LEAD}" != 0 ]; then name="${name}_lead${D_LEAD}"; fi
  if [ -n "${NO_FORCE:-}" ]; then
    name="${name}_nf"
    if [ -n "${NO_FORCE_TARGET:-}" ]; then name="${name}t"; fi
  elif [ -n "${NO_FORCE_TARGET:-}" ]; then name="${name}_nft"; fi
  if [ -n "${SURPRISE:-}" ]; then name="${name}_sur"; fi
  if [ -n "${NO_EVAL:-}" ]; then
    printf '%-18s train %s\n' "$name" "$tid"
    continue
  fi
  # aftercorr pairs the arrays task by task, so eval seed k waits only for train seed k
  eid=$(NAME="$name" sbatch --parsable ${ARRAY:+--array="$ARRAY"} \
        --dependency=aftercorr:"$tid" policy/slurm/eval_policy.sbatch)
  eval_ids+=("$eid")
  printf '%-18s train %-9s eval %s\n' "$name" "$tid" "$eid"
done

if [ "${#eval_ids[@]}" -gt 0 ]; then
  dep=$(IFS=:; printf '%s' "${eval_ids[*]}")
  cid=$(sbatch --parsable --dependency=afterany:"$dep" policy/slurm/compare.sbatch)
  printf '%-14s %s  -> policy/runs/COMPARISON.md\n' compare "$cid"
fi
echo "logs in $(pwd)/logs; checkpoints and results in $(pwd)/policy/runs/<name>/seed<k>/"
