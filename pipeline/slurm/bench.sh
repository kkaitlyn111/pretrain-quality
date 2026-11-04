#!/bin/bash
# HellaSwag / ARC-Easy / LAMBADA for validation checkpoint V<array index>.
#   OUT_SUFFIX=_60m_2ep sbatch --array=0-9 pipeline/slurm/bench.sh
#SBATCH --job-name=pq_bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
source pipeline/slurm/_env.sh
: "${OUT_SUFFIX:=_60m_2ep}"
RID="V${SLURM_ARRAY_TASK_ID}"
OUT="$PQ_WORK_DIR/validation_results_eq${OUT_SUFFIX}"
$PY pipeline/scripts/12_eval_benchmarks.py --ckpt "$OUT/${RID}_ckpt.pt" --rid "$RID" --out-dir "$OUT"
