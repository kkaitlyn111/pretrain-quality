# Sourced by every SLURM script: load site config, set common env, define $PY.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
# shellcheck disable=SC1090
source "${PQ_ENV:-$ROOT/config/cluster.env}"
export TOKENIZERS_PARALLELISM=false
export WANDB_SILENT=true
PY="${PQ_PYTHON:-python}"
mkdir -p "${PQ_LOGS_DIR:-logs}"
shard_at() {  # shard_at <json> <index>  -> shard id
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['shards'][int(sys.argv[2])])" "$1" "$2"
}
