"""Perplexity-correlation fastText filters: name -> (Hugging Face repo, file)."""

from __future__ import annotations
from pathlib import Path

# Patch fasttext's NumPy-2 incompatibility before it is used.
import pretrain_quality.features._fasttext_patch  # noqa: F401,E402

from huggingface_hub import hf_hub_download

from config import MODELS_DIR

PC_FASTTEXT_MODELS = {
    "arc_easy":   ("perplexity-correlations/fasttext-arc-easy-target",   "arc_easy_target.bin"),
    "piqa":       ("perplexity-correlations/fasttext-piqa-target",       "piqa_target.bin"),
    "sciq":       ("perplexity-correlations/fasttext-sciq-target",       "sciq_target.bin"),
    "lambada":    ("perplexity-correlations/fasttext-lambada-target",    "lambada_target.bin"),
}


def ensure_model(name: str) -> Path:
    repo_id, fn = PC_FASTTEXT_MODELS[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(repo_id=repo_id, filename=fn, local_dir=str(MODELS_DIR / name))
    return Path(local)
