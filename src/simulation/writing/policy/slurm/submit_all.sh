#!/bin/bash
# Submit all four structures, three seeds each (12 training jobs).
# Run from anywhere inside the uploaded repo:  bash policy/slurm/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> src/simulation/writing, which becomes SLURM_SUBMIT_DIR
mkdir -p logs
for s in a_ft_input b_uni_dir c_unified d_cross_cond; do
  sbatch "policy/slurm/train_${s}.sbatch"
done
echo "logs in $(pwd)/logs; checkpoints in $(pwd)/policy/runs/<name>/seed<k>/"
