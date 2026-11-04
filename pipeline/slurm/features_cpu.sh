#!/bin/bash
# CPU features for one shard per array task: heuristics, synth heuristics,
# textbook-quality fastText, perplexity-corr fastText, spaCy entities.
#   sbatch -p "$PQ_CPU_PARTITION" --array=0-99%30 pipeline/slurm/features_cpu.sh [shards.json]
#SBATCH --job-name=pq_feat_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
SHARDS_JSON="${1:-$PQ_FEATURES_DIR/sampled_shards.json}"
SHARD=$(shard_at "$SHARDS_JSON" "$SLURM_ARRAY_TASK_ID")
echo "=== task=$SLURM_ARRAY_TASK_ID shard=$SHARD on $(hostname) ==="
cd pipeline/features
$PY 02_extract_heuristics.py     --shards "$SHARD" --n-workers 4
$PY 03_extract_synth_features.py --shards "$SHARD" --n-workers 4
$PY 05_score_textbook_quality.py --shards "$SHARD" --n-workers 2
$PY 08_score_perplexity_corr.py  --model all --shards "$SHARD" --n-workers 2
$PY 04_extract_entities_batched.py --shard "$SHARD" --batch-size 128
