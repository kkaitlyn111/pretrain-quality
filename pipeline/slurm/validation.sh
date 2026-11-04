#!/bin/bash
# Equal-unique-token validation run V<array index> on the fresh pool; saves a checkpoint.
#   MODEL_PRESET=60m TOTAL_TOKENS=150000000 EPOCHS=2 OUT_SUFFIX=_60m_2ep \
#       sbatch --array=0-9 pipeline/slurm/validation.sh
#SBATCH --job-name=pq_val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
: "${MODEL_PRESET:=15m}"
: "${TOTAL_TOKENS:=150000000}"
: "${EPOCHS:=1}"
: "${POOL:=orig}"
: "${OUT_SUFFIX:=_${MODEL_PRESET}_${EPOCHS}ep_$((TOTAL_TOKENS/1000000))M}"
RID="V${SLURM_ARRAY_TASK_ID}"
echo "=== val $RID pool=$POOL preset=$MODEL_PRESET unique=$TOTAL_TOKENS epochs=$EPOCHS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY pipeline/scripts/11_run_validation.py \
    --run "$RID" --budget "$TOTAL_TOKENS" --pool "$POOL" --epochs "$EPOCHS" \
    --model-preset "$MODEL_PRESET" --save-checkpoint --out-suffix "$OUT_SUFFIX" \
    --wandb-project "${WANDB_PROJECT:-pq-validation}"
