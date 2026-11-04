"""Model-scored features over a list of texts (used by `pq extract --recompute`).

Each scorer mirrors the corresponding per-shard script in pipeline/features/:

  entities          04_extract_entities.py       spaCy en_core_web_sm, CPU (*)
  textbook_quality  05_score_textbook_quality.py fastText, CPU (downloads ~GBs of weights)
  ai_generated      06_score_ai_detector.py      RoBERTa-base, GPU recommended
  qurater           07_score_qurater.py          QuRater-1.3B, GPU strongly recommended

(*) entity_density for documents longer than 10k characters: most training shards were
    scored with 04_extract_entities_batched.py, which divides by the word count of the
    truncated text rather than the full document, so a recompute differs on those docs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from pretrain_quality import paths

# ---------------------------------------------------------------- entities

ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "NORP", "FAC", "DATE", "QUANTITY",
}
ENTITY_MAX_CHARS = 10_000


def load_spacy():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def entity_features(nlp, text: str) -> tuple[float, float, float]:
    """(entity_density per 1k words, unique/mentions, unique per sentence)."""
    t = text[:ENTITY_MAX_CHARS]
    if not t.strip():
        return float("nan"), float("nan"), float("nan")
    doc = nlp(t)
    ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]
    n_unique = len({e.text.lower() for e in ents})
    n_sents = max(sum(1 for _ in doc.sents), 1)
    # density uses the FULL doc's word count so long docs compare fairly
    density = n_unique / (max(len(text.split()), 1) / 1000.0)
    diversity = (n_unique / len(ents)) if ents else 0.0
    return density, diversity, n_unique / n_sents


def score_entities(texts: list[str]) -> np.ndarray:
    nlp = load_spacy()
    return np.asarray([entity_features(nlp, t) for t in tqdm(texts, desc="entities")],
                      dtype=np.float32)


# ---------------------------------------------------------------- textbook quality

TEXTBOOK_REPO = "kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2"
_HIGH = {"__label__High", "__label__high"}
_MID = {"__label__Mid", "__label__mid", "__label__Medium", "__label__medium"}


def load_fasttext(repo_id: str, filename: str, subdir: str):
    from huggingface_hub import hf_hub_download

    from ._fasttext_patch import fasttext
    local = hf_hub_download(repo_id=repo_id, filename=filename,
                            local_dir=str(paths.MODELS_DIR / subdir))
    return fasttext.load_model(str(Path(local)))


def score_textbook(texts: list[str]) -> np.ndarray:
    """P(High) + 0.5 * P(Mid) from the textbook-quality fastText classifier."""
    model = load_fasttext(TEXTBOOK_REPO, "model.bin", "textbook_quality")
    out = []
    for text in tqdm(texts, desc="textbook_quality"):
        text = text.replace("\n", " ").replace("\r", " ").strip()
        if not text:
            out.append(float("nan"))
            continue
        labels, probs = model.predict(text, k=-1)
        p = dict(zip(labels, map(float, probs)))
        out.append(sum(v for k, v in p.items() if k in _HIGH)
                   + 0.5 * sum(v for k, v in p.items() if k in _MID))
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------- transformer scorers

def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _chunked_scores(texts, tok, run_batch, ctx: int, batch_size: int, n_out: int,
                    desc: str) -> np.ndarray:
    """Split each doc into `ctx`-token windows, score every window, and return the
    window-length-weighted mean per doc (same scheme as the per-shard scripts)."""
    flat = []  # (doc, ids)
    for d, text in enumerate(texts):
        ids = tok(text or "", add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            ids = [0]
        flat += [(d, ids[i:i + ctx]) for i in range(0, len(ids), ctx)]
    flat.sort(key=lambda it: len(it[1]))  # length-sorted batches = less padding
    sums = np.zeros((len(texts), n_out))
    weights = np.zeros(len(texts))
    for start in tqdm(range(0, len(flat), batch_size), desc=desc):
        items = flat[start:start + batch_size]
        vals = run_batch([ids for _, ids in items])
        for (d, ids), v in zip(items, vals):
            sums[d] += v * len(ids)
            weights[d] += len(ids)
    return (sums / np.maximum(weights, 1)[:, None]).astype(np.float32)


def score_qurater(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """princeton-nlp/QuRater-1.3B logits: writing_style, required_expertise,
    facts_and_trivia, educational_value (512-token windows)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo, dev = "princeton-nlp/QuRater-1.3B", _device()
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model.config.pad_token_id = 0
    model.eval().to(dev)

    @torch.inference_mode()
    def run(batch):
        L = max(map(len, batch))
        ids = torch.zeros((len(batch), L), dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, b in enumerate(batch):
            ids[i, :len(b)] = torch.tensor(b)
            attn[i, :len(b)] = 1
        return model(input_ids=ids.to(dev), attention_mask=attn.to(dev)).logits.float().cpu().numpy()

    return _chunked_scores(texts, tok, run, 512, batch_size, 4, "qurater")


def score_ai_detector(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """P(machine-generated) from openai-community/roberta-base-openai-detector."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo, dev = "openai-community/roberta-base-openai-detector", _device()
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    model.eval().to(dev)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    fake = next((i for i, n in id2label.items() if "fake" in n), 1)
    cls = tok.cls_token_id if tok.cls_token_id is not None else 0
    sep = tok.sep_token_id if tok.sep_token_id is not None else 2
    pad = tok.pad_token_id if tok.pad_token_id is not None else 1

    @torch.inference_mode()
    def run(batch):
        L = max(map(len, batch)) + 2
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for i, b in enumerate(batch):
            ids[i, 0], ids[i, 1:1 + len(b)], ids[i, 1 + len(b)] = cls, torch.tensor(b), sep
            attn[i, :2 + len(b)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=attn.to(dev)).logits
        return torch.softmax(logits.float(), dim=-1)[:, fake:fake + 1].cpu().numpy()

    return _chunked_scores(texts, tok, run, 510, batch_size, 1, "ai_detector")[:, 0]
