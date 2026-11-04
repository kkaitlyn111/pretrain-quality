#!/bin/bash
# Experimental: influence scores for one data partition per array task (resumable).
#   N=64; sbatch --array=0-$((N-1)) pipeline/slurm/influence_score.sh $N
#SBATCH --job-name=pq_score_infl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
N_PART=${1:-64}
$PY pipeline/influence/scripts/03_score_influence.py --ckpt "$PQ_WORK_DIR/models/full_pool_100m.pt" \
    --factors-name ekfac_100m --scores-name doc_scores \
    --n-partitions "$N_PART" --partition-id "$SLURM_ARRAY_TASK_ID" \
    --per-device-train-batch-size 8 --per-device-query-batch-size 8
