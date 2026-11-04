#!/bin/bash
# Experimental: fit EK-FAC factors (one 80GB GPU).
#SBATCH --job-name=pq_fit_factors
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
source pipeline/slurm/_env.sh
$PY pipeline/influence/scripts/02_fit_factors.py --ckpt "$PQ_WORK_DIR/models/full_pool_100m.pt" \
    --factors-name ekfac_100m --n-fit 50000 --per-device-batch-size 8
