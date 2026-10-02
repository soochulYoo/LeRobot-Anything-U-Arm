#!/bin/bash
# Step 2 of PEG_DEMO_PLAN.md: does ONE constant stiffness do the whole job?
#
# Four sub-sweeps, because the knobs are coupled in Case 1 and a single grid
# would confound them:
#   A  axial stiffness at a HELD press (depth = press/k_axial, catch = 40% of it)
#   B  press at a held axial stiffness
#   C  lateral stiffness
#   D  K_R
# Three aim errors each: 0 (the hole is where it looks), 4 (where the cascade
# jammed), 8 (the most the cascade's scrub recovered).
set -u
PY=/scratch2/soochul/venvs/writingsim/bin/python
E=${1:-0,4,8}
N=${2:-4}
D=logs/peg2
mkdir -p $D
common="--errors $E --episodes $N --catch-frac 0.4"
for k in 400 800 1600 3000; do
  nohup $PY peg_insertion_case1.py $common --press 8 --k-axial-search $k \
    > $D/A_kax$k.log 2>&1 &
done
for f in 4 8 16 24; do
  nohup $PY peg_insertion_case1.py $common --press $f --k-axial-search 800 \
    > $D/B_press$f.log 2>&1 &
done
for kl in 150 300 800 2000; do
  nohup $PY peg_insertion_case1.py $common --press 8 --k-lat $kl \
    > $D/C_klat$kl.log 2>&1 &
done
for kr in 2 10 40 80; do
  nohup $PY peg_insertion_case1.py $common --press 8 --kr-search $kr \
    > $D/D_kr$kr.log 2>&1 &
done
wait
