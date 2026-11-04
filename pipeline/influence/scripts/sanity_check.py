"""Sanity-check the EK-FAC influence scores.

(1) Distribution + what raw influence correlates with (length artifact? existing
    quality filters?). (2) Eyeball the actual text of the top/bottom-ranked docs.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from config import FEATURES_CODE, M2_DATA, CORPUS_DIR  # noqa: E402
from config import MASTER_PARQUET, DATA_DIR, REPO_ROOT  # noqa

import importlib.util as _ilu
def _load(name, p):
    spec = _ilu.spec_from_file_location(name, str(p)); m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m); return m
_m2cfg = _load("m2cfg", FEATURES_CODE / "config.py")
_m2io = _load("m2io", FEATURES_CODE / "_shared_io.py")


def main():
    infl = pd.read_parquet(DATA_DIR / "influence_scores.parquet")
    raw = infl["raw_influence"].to_numpy().astype(np.float64)
    print(f"n = {len(infl):,}")
    print(f"raw influence: mean={raw.mean():.3e} median={np.median(raw):.3e} "
          f"std={raw.std():.3e}  %positive={100*(raw>0).mean():.1f}")

    master = pd.read_parquet(MASTER_PARQUET)
    tr = master[master["__in_train_pool"]].reset_index(drop=True)
    assert len(tr) == len(infl), (len(tr), len(infl))
    # verify alignment (influence is in train-pool row order)
    same = (tr["__shard_idx"].to_numpy()[:1000] == infl["shard_idx"].to_numpy()[:1000]).all()
    print(f"alignment (first 1000 shard ids match): {same}")

    print("\n=== Spearman(raw_influence, feature): what is influence tracking? ===")
    feats = ["token_count", "char_count", "word_count", "ngram_rep_2", "type_token_ratio",
             "dclm_score", "fwedu_rounded", "qr_educational_value", "qr_required_expertise",
             "textbook_quality", "ai_generated", "entity_density", "flesch_reading_ease"]
    for f in feats:
        if f in tr.columns:
            v = tr[f].to_numpy()
            m = np.isfinite(v)
            rho, _ = spearmanr(raw[m], v[m])
            print(f"  {f:22s} rho = {rho:+.3f}")

    # also |influence| vs length (magnitude artifact check)
    tok = tr["token_count"].to_numpy().astype(float)
    rho_abs, _ = spearmanr(np.abs(raw), tok)
    print(f"  {'|influence| vs token_count':22s} rho = {rho_abs:+.3f}   (>0.7 => length-dominated magnitude)")

    # ---- top/bottom docs by raw influence ----
    K = 8
    order = np.argsort(raw)
    pick = {"TOP (highest raw)": order[-K:][::-1], "BOTTOM (lowest raw)": order[:K]}
    # group needed docs by shard for one decompression pass each
    want = {}
    for label, idxs in pick.items():
        for r in idxs:
            s = int(infl["shard_idx"].iloc[r]); d = int(infl["doc_idx"].iloc[r])
            want.setdefault(s, {})[d] = (label, r)
    text_by_rowidx = {}
    for s, dmap in want.items():
        dp = _m2cfg.doc_path(s)
        hi = max(dmap)
        for j, doc in _m2io.iter_docs(dp):   # iter_docs yields (line_idx, doc_dict)
            if j in dmap:
                text_by_rowidx[dmap[j][1]] = (doc.get("text", ""), doc.get("url", ""))
            if j > hi:
                break

    for label, idxs in pick.items():
        print(f"\n========== {label} ==========")
        for r in idxs:
            s = int(infl["shard_idx"].iloc[r]); d = int(infl["doc_idx"].iloc[r])
            tkn = int(tr["token_count"].iloc[r]) if "token_count" in tr.columns else -1
            dclm = float(tr["dclm_score"].iloc[r]); fwe = float(tr["fwedu_rounded"].iloc[r])
            topic = tr["topic"].iloc[r]
            txt, url = text_by_rowidx.get(r, ("<text n/a>", ""))
            snip = " ".join(txt.split())[:240]
            print(f"\n  raw={raw[r]:+.3e} tok={tkn} dclm={dclm:.2f} fwedu={fwe:.0f} topic={topic}")
            print(f"  url={url[:80]}")
            print(f"  text: {snip}")


if __name__ == "__main__":
    main()
