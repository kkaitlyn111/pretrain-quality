"""Shared paths for the swarm / target / EBM / validation pipeline.

Every location derives from pretrain_quality.paths, i.e. the PQ_* environment variables
(see config/cluster.env.example). Nothing cluster-specific is hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pretrain_quality import paths

REPO_ROOT = paths.REPO_ROOT
PIPELINE_ROOT = Path(__file__).resolve().parent
FEATURES_CODE = PIPELINE_ROOT / "features"   # per-shard feature-extraction scripts
M2_DATA = paths.FEATURES_DIR                 # per-shard feature outputs + annotations.parquet
CORPUS_DIR = paths.CORPUS_DIR                # WebOrganizer/Corpus-200B checkout

# Outputs
DATA_DIR = paths.WORK_DIR
TOKENS_DIR = DATA_DIR / "tokens"
MASKS_DIR = DATA_DIR / "selection_masks"
RESULTS_DIR = DATA_DIR / "diagnostic_results"
LOGS_DIR = paths.LOGS_DIR
CONFIGS_DIR = DATA_DIR / "configs"
MODELS_DIR = DATA_DIR / "models"

MASTER_PARQUET = DATA_DIR / "master.parquet"
VAL_SHARDS_JSON = DATA_DIR / "val_shard_indices.json"
DIAGNOSTIC_CONFIGS_JSON = DATA_DIR / "diagnostic_run_configs.json"

# Pretokenization outputs
TRAIN_TOKEN_IDS = TOKENS_DIR / "train_pool_token_ids.bin"
TRAIN_DOC_OFFSETS = TOKENS_DIR / "train_pool_doc_offsets.npy"
TRAIN_DOC_INDEX = TOKENS_DIR / "train_pool_doc_index.npy"  # global doc idx per token-row
VAL_TOKEN_IDS = TOKENS_DIR / "val_token_ids.bin"
VAL_DOC_OFFSETS = TOKENS_DIR / "val_doc_offsets.npy"
VAL_SEQUENCES = TOKENS_DIR / "val_sequences_1024.npy"
VAL_SEQUENCES_EVAL = TOKENS_DIR / "val_sequences_eval_1024.npy"  # subsample for fast eval
TOKENS_META = TOKENS_DIR / "tokens_meta.json"

# Per-shard pretokenized files
PERSHARD_TOKENS_DIR = TOKENS_DIR / "pershard"

# Constants
SEED = 42
SEQ_LEN = 1024
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
N_LM_VAL_SHARDS = 10

# Feature-extraction helpers (config/_shared_io of the features stage) are
# loaded by file path in the scripts; keep their directory importable too.
sys.path.append(str(FEATURES_CODE))

for d in (DATA_DIR, TOKENS_DIR, MASKS_DIR, RESULTS_DIR, LOGS_DIR, CONFIGS_DIR,
          MODELS_DIR, PERSHARD_TOKENS_DIR):
    d.mkdir(parents=True, exist_ok=True)
