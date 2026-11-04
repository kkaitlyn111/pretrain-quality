"""Corpus / per-shard feature locations for the feature-extraction scripts.

Paths come from pretrain_quality.paths (PQ_* environment variables); nothing is hardcoded.
"""
from __future__ import annotations

from pathlib import Path

from pretrain_quality import paths

SEED = 42
N_SHARDS = 100
TOTAL_SHARDS = 9888

REPO_ROOT = paths.REPO_ROOT
CORPUS_ROOT = paths.CORPUS_DIR
DATA_ROOT = paths.FEATURES_DIR
FEATURES_DIR = DATA_ROOT / "features"
MODELS_DIR = paths.MODELS_DIR

SAMPLED_SHARDS_JSON = DATA_ROOT / "sampled_shards.json"
ANNOTATIONS_PARQUET = DATA_ROOT / "annotations.parquet"

# Per-subdir filename conventions. Each entry maps a logical "key" to
# (subdir_name, suffix). Files are named "{stem}{suffix}.npy" where
# stem = CC_shard_NNNNNNNN_processed.
ANNOT_SPECS = {
    "tokens":        ("tokens",                       ""),
    "dclm_score":    ("scores_dclm-fasttext",         ""),
    "fwedu_score":   ("scores_fineweb-edu",           ""),
    "fwedu_rounded": ("scores_fineweb-edu__rounded",  "__rounded"),
    "topic":         ("domains_topics",               "__choice"),
    "topic_logits":  ("domains_topics__logits",       "__logits"),
    "format":        ("domains_formats",              "__choice"),
    "format_logits": ("domains_formats__logits",      "__logits"),
    "cluster24":     ("domains_clusters-k24",         ""),
}

DOC_SUBDIR = "documents"
DOC_SUFFIX = ".jsonl.zst"


def shard_stem(shard_idx: int) -> str:
    return f"CC_shard_{shard_idx:08d}_processed"


def annot_path(shard_idx: int, key: str) -> Path:
    subdir, suffix = ANNOT_SPECS[key]
    return CORPUS_ROOT / subdir / f"{shard_stem(shard_idx)}{suffix}.npy"


def doc_path(shard_idx: int) -> Path:
    return CORPUS_ROOT / DOC_SUBDIR / f"{shard_stem(shard_idx)}{DOC_SUFFIX}"
