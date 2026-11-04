#!/bin/bash
# One 15.4M-param diagnostic LM per array task (run_id = array index), trained on
# exactly 150M unique tokens of that run's selection; writes val_loss + mask.
#   sbatch --array=0-790%35 pipeline/slurm/diagnostic.sh
#SBATCH --job-name=pq_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
echo "=== run_id=$SLURM_ARRAY_TASK_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY pipeline/scripts/run_diagnostic.py --run-id "$SLURM_ARRAY_TASK_ID" \
    --configs "$PQ_WORK_DIR/diagnostic_run_configs_trimmed.json" \
    --total-tokens 150000000 --unique-tokens 150000000 \
    --wandb-project "${WANDB_PROJECT:-pq-diagnostic}"
