#!/bin/bash
# Experimental: aggregate influence -> target -> refit EBM (tag 'influence') -> diagnostics.
#SBATCH --job-name=pq_infl_final
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source pipeline/slurm/_env.sh
W="$PQ_WORK_DIR"
$PY pipeline/influence/scripts/04_aggregate_influence.py --scores-name doc_scores
$PY pipeline/scripts/07_train_ebm.py --target "$W/document_quality_target_influence.npy" --out-tag influence
$PY pipeline/influence/scripts/05_influence_diagnostics.py \
    --influence "$W/document_quality_target_influence.npy" --dim "$W/document_quality_target.npy" \
    --ebm-dim "$W/ebm_summary.json" --ebm-infl "$W/ebm_summary_influence.json"
