"""All filesystem locations, read from environment variables.

Nothing cluster-specific is hardcoded; copy config/cluster.env.example to
config/cluster.env, edit it, and `source` it (the SLURM scripts do this).
Defaults point inside the repo so the quickstart works with no setup.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


# Raw WebOrganizer/Corpus-200B checkout (documents/, domains_*, scores_*, tokens/).
CORPUS_DIR = _env_path("PQ_CORPUS_DIR", REPO_ROOT / "data" / "Corpus-200B")
# Per-shard feature outputs + annotations.parquet + sampled_shards.json.
FEATURES_DIR = _env_path("PQ_FEATURES_DIR", REPO_ROOT / "data" / "features")
# Pipeline outputs: master.parquet, diagnostic runs, targets, EBMs, validation runs.
WORK_DIR = _env_path("PQ_WORK_DIR", REPO_ROOT / "data" / "pipeline")
# Downloaded classifier weights (fastText perplexity-correlation, textbook-quality).
MODELS_DIR = _env_path("PQ_MODELS_DIR", REPO_ROOT / "data" / "models")
LOGS_DIR = _env_path("PQ_LOGS_DIR", REPO_ROOT / "logs")
