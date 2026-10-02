#!/bin/bash
# Step 2: with the search motion fixed, does ONE constant stiffness do the job?
#
# The scrub has two conflicting demands.  Finding the hole wants a soft lateral
# wrist, so the tip can be dragged across the face and fall in without the arm
# fighting it.  Going home afterwards wants enough lateral authority to break a
# wedge instead of pinning the peg at 30 N.  If one (ratio, rot_ratio) serves
# both, there is nothing for a learned stiffness to do and step 3 is pointless.
#
#   ./peg_stiffness_sweep.sh <errors> <episodes>
set -u
PY=/scratch2/soochul/venvs/writingsim/bin/python
ERRS=${1:-4,8}
EPS=${2:-6}
mkdir -p logs/peg
for ratio in 3 10 30 100; do
  for rr in 1 10; do
    nohup $PY peg_search_study.py --errors "$ERRS" --episodes "$EPS" --push 0 \
      --ratio $ratio --rot-ratio $rr > logs/peg/stiff_r${ratio}_rr${rr}.log 2>&1 &
  done
done
wait
