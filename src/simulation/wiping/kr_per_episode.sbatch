#!/bin/bash
#SBATCH --job-name=kr_pe
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --array=0-23
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kr_pe_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "/scratch2/soochul/venvs/writingsim/bin/activate"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PAD=(0.040 0.040 0.015); RAD=(0.15 0.10 0.25)
i=$SLURM_ARRAY_TASK_ID
c=$(( i / 8 )); s=$(( i % 8 ))
export WIPE_PAD_R="${PAD[$c]}"
echo "### pad ${PAD[$c]} R ${RAD[$c]} seed ${s}"
python -u kr_per_episode.py --radius "${RAD[$c]}" --seed "$s"
