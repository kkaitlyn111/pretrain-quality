"""Batched variant of 04_extract_entities.py using spaCy's nlp.pipe (one shard per call).

Usage:
    python 04_extract_entities_batched.py --shard 431
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import DATA_ROOT, shard_stem, doc_path  # noqa
from pretrain_quality.io import iter_docs  # noqa

OUT_DIR = DATA_ROOT / "entities"
ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "NORP", "FAC", "DATE", "QUANTITY",
}
MAX_CHARS = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = OUT_DIR / f"{shard_stem(args.shard)}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"already exists: {out_path}")
        return

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    t0 = time.time()
    texts = []
    word_counts = []
    for _i, obj in iter_docs(doc_path(args.shard)):
        text = (obj.get("text", "") or "")[:MAX_CHARS]
        # NB: word count of the *truncated* text, unlike 04_extract_entities.py, which
        # uses the full document. The two agree for docs <= MAX_CHARS; in the reported
        # run 64 of 90 training shards came from this script and 26 from 04.
        word_counts.append(len(text.split()))
        texts.append(text)
    n_docs = len(texts)
    print(f"  loaded {n_docs} docs in {time.time()-t0:.1f}s, running spaCy...")

    t1 = time.time()
    dens = np.zeros(n_docs, dtype=np.float32)
    divr = np.zeros(n_docs, dtype=np.float32)
    persent = np.zeros(n_docs, dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=args.batch_size)):
        ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]
        n_men = len(ents)
        uniq = {e.text.lower() for e in ents}
        n_uniq = len(uniq)
        n_sents = max(sum(1 for _ in doc.sents), 1)
        wc = word_counts[i]
        dens[i] = n_uniq / (max(wc, 1) / 1000.0)
        divr[i] = (n_uniq / n_men) if n_men else 0.0
        persent[i] = n_uniq / n_sents
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_docs} ({(i+1)/(time.time()-t1):.1f} docs/s)")
    print(f"  spaCy done in {time.time()-t1:.1f}s, {n_docs/(time.time()-t1):.1f} docs/s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, entity_density=dens, entity_diversity=divr,
             entity_per_sentence=persent)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
