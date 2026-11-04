#!/bin/bash
# GPT-NeoX-tokenize one shard per array task.
#   sbatch -p "$PQ_CPU_PARTITION" --array=0-99%30 pipeline/slurm/pretokenize.sh [shards.json]
#SBATCH --job-name=pq_pretok
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
SHARDS_JSON="${1:-$PQ_FEATURES_DIR/sampled_shards.json}"
SHARD=$(shard_at "$SHARDS_JSON" "$SLURM_ARRAY_TASK_ID")
echo "Array task $SLURM_ARRAY_TASK_ID -> shard $SHARD"
$PY pipeline/scripts/01_pretokenize_shard.py --shard "$SHARD"
