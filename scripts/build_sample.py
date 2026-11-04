"""Build the bundled quickstart sample (cluster-only; needs the full pipeline outputs).

Draws N documents from the *held-out test split* of the EBM fit
(the exact 80/20 split, stratified by topic, seed 42, used by
pipeline/scripts/07_train_ebm.py), so that the bundled full-data EBM can be
evaluated honestly on the sample.

Writes:
  data/sample/docs.parquet          text + WebOrganizer topic/format + model-scored
                                    features (QuRater, textbook, AI-detector, spaCy
                                    entities, perplexity-corr) + target y
  data/sample/reference.parquet     the original pipeline's CPU-heuristic values and
                                    the full-data EBM-no-PC prediction, for checking
  results/feature_medians.json      train-pool medians used for NaN imputation
  models/ebm_no_pc.joblib           the full-data EBM-no-PC (copied)

Usage (paths come from PQ_* env vars, see config/cluster.env.example):
  python scripts/build_sample.py --n 10000
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd
from sklearn.model_selection import train_test_split

from pretrain_quality import paths
from pretrain_quality.features.columns import (CAT_FEATURES, CPU_FEATURES, MODEL_FEATURES,
                                  NO_PC_FEATURES, PC_FEATURES)

REPO = Path(__file__).resolve().parents[1]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")


def _read_shard(args):
    shard, wanted = args
    path = paths.CORPUS_DIR / "documents" / f"CC_shard_{shard:08d}_processed.jsonl.zst"
    wanted = set(wanted)
    out = {}
    with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
        for i, line in enumerate(io.TextIOWrapper(r, encoding="utf-8")):
            if i in wanted:
                out[i] = json.loads(line).get("text", "") or ""
                if len(out) == len(wanted):
                    break
    return shard, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    master = pd.read_parquet(paths.WORK_DIR / "master.parquet")
    is_train = master["__in_train_pool"].to_numpy()
    df = master[is_train].reset_index(drop=True)
    y = np.load(paths.WORK_DIR / "document_quality_target.npy")
    pred = np.load(paths.WORK_DIR / "ebm_interp_predicted_scores.npy")[is_train]
    assert len(y) == len(df) == len(pred)

    # Reproduce the exact EBM train/test split.
    idx = np.arange(len(df))
    _, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                 stratify=df["topic"].to_numpy())
    yt, pt = y[te_idx], pred[te_idx]
    r2_full_test = 1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    print(f"full test split: n={len(te_idx):,}  EBM-no-PC R^2={r2_full_test:.4f}")

    medians = {c: float(df[c].median()) for c in CPU_FEATURES + MODEL_FEATURES + PC_FEATURES + ["token_count"]}

    # Oversample, then drop documents containing e-mail addresses (public web text, but
    # no reason to redistribute contact details), keeping the first n in random order.
    rng = np.random.default_rng(args.seed)
    cand = rng.choice(te_idx, size=int(args.n * 1.1), replace=False)
    by_shard = defaultdict(list)
    for i in cand:
        by_shard[int(df.at[i, "__shard_idx"])].append(int(df.at[i, "__doc_idx"]))
    with Pool(args.workers) as pool:
        texts = dict(pool.map(_read_shard, list(by_shard.items())))
    cand_text = [texts[int(df.at[i, "__shard_idx"])][int(df.at[i, "__doc_idx"])] for i in cand]
    keep = [k for k, t in enumerate(cand_text) if not EMAIL_RE.search(t)][:args.n]
    assert len(keep) == args.n, "not enough e-mail-free candidates; raise the oversampling"
    print(f"dropped {len(cand) - len([t for t in cand_text if not EMAIL_RE.search(t)])} "
          f"of {len(cand)} candidates containing e-mail addresses")
    order = np.argsort(cand[keep])
    pick = cand[keep][order]
    sub = df.iloc[pick].reset_index(drop=True)
    sub["text"] = [cand_text[keep[j]] for j in order]

    docs = pd.DataFrame({
        "doc_id": [f"{int(s):05d}-{int(d):06d}" for s, d in zip(sub["__shard_idx"], sub["__doc_idx"])],
        "text": sub["text"],
        "token_count": sub["token_count"].astype("int32"),
        "topic": sub["topic"].astype("int8"),
        "format": sub["format"].astype("int8"),
    })
    for c in MODEL_FEATURES + PC_FEATURES:
        docs[c] = sub[c].astype("float32")
    docs["y"] = y[pick].astype("float32")

    ref = pd.DataFrame({"doc_id": docs["doc_id"]})
    for c in CPU_FEATURES:
        ref[c] = sub[c].astype("float64")
    ref["ebm_no_pc_full_pred"] = pred[pick].astype("float32")

    out = REPO / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out / "docs.parquet", index=False, compression="zstd", compression_level=19)
    ref.to_parquet(out / "reference.parquet", index=False, compression="zstd")
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "feature_medians.json").write_text(json.dumps(medians, indent=2) + "\n")
    (REPO / "models").mkdir(exist_ok=True)
    shutil.copy(paths.WORK_DIR / "models" / "ebm_interp.joblib", REPO / "models" / "ebm_no_pc.joblib")

    ys, ps = docs["y"].to_numpy(), ref["ebm_no_pc_full_pred"].to_numpy()
    r2_sample = 1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    print(f"sample: n={len(docs):,}  EBM-no-PC R^2={r2_sample:.4f}")
    print(f"features used by EBM-no-PC: {len(NO_PC_FEATURES)} continuous + {len(CAT_FEATURES)} categorical")
    for f in (out / "docs.parquet", out / "reference.parquet"):
        print(f"wrote {f.relative_to(REPO)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
