#!/bin/bash
# GPU features for one shard per array task: QuRater-1.3B (4 axes) + RoBERTa AI detector.
#   sbatch --array=0-99%20 pipeline/slurm/features_gpu.sh [shards.json]
#SBATCH --job-name=pq_feat_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
SHARDS_JSON="${1:-$PQ_FEATURES_DIR/sampled_shards.json}"
SHARD=$(shard_at "$SHARDS_JSON" "$SLURM_ARRAY_TASK_ID")
echo "=== task=$SLURM_ARRAY_TASK_ID shard=$SHARD on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
cd pipeline/features
$PY 07_score_qurater.py     --shards "$SHARD" --batch-size 32
$PY 06_score_ai_detector.py --shards "$SHARD" --batch-size 64
