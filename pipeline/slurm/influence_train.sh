#!/bin/bash
# Experimental: train the ~100M-param attribution model on the full pool (1 epoch).
#SBATCH --job-name=pq_fullpool
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
source pipeline/slurm/_env.sh
$PY pipeline/influence/scripts/01_train_attribution_model.py --preset 100m --epochs 1.0 \
    --batch-size 256 --micro-batch-size 32 --lr 2e-3 \
    --wandb-project "${WANDB_PROJECT:-pq-fullpool}" --run-name fullpool_100m_1ep
