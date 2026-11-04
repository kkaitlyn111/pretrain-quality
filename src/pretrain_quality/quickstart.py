"""Quickstart: feature extraction -> EBM fit on the bundled 10k-document sample.

  pq extract   text -> 34 CPU heuristic features (+ optional model features)
  pq fit       Lasso / EBM-no-PC on an 80/20 split of the sample, plus the
                bundled full-data EBM scored on the same (held-out) documents
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from pretrain_quality import paths
from pretrain_quality.features.columns import (CAT_FEATURES, CPU_FEATURES, MODEL_FEATURES,
                                  NO_PC_FEATURES, QURATER_FEATURES, ENTITY_FEATURES)

SAMPLE_DIR = paths.REPO_ROOT / "data" / "sample"
RESULTS_DIR = paths.REPO_ROOT / "results"
RUNS_DIR = paths.REPO_ROOT / "runs"
RECOMPUTABLE = ("entities", "textbook", "ai", "qurater")


# ------------------------------------------------------------------ extract

def _cpu_features(text: str) -> dict:
    from pretrain_quality.features.heuristics import extract_features
    from pretrain_quality.features.synth import extract_synth_features
    out = extract_features(text, "")
    out.update(extract_synth_features(text))
    return {k: out[k] for k in CPU_FEATURES}


def extract(args) -> None:
    docs = pd.read_parquet(args.docs)
    if args.limit:
        docs = docs.head(args.limit)
    texts = docs["text"].tolist()
    print(f"[extract] {len(docs):,} docs from {args.docs}")

    from pretrain_quality.features.synth import ensure_cmudict
    ensure_cmudict()  # fetch once before forking workers
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(_cpu_features, texts, chunksize=64)
    feats = pd.DataFrame(rows, columns=CPU_FEATURES)
    print(f"[extract] {len(CPU_FEATURES)} CPU features in {time.time() - t0:.1f}s "
          f"({args.workers} workers)")

    out = pd.concat([docs[["doc_id", "topic", "format", "token_count", "y"]].reset_index(drop=True),
                     feats], axis=1)
    # Model-scored features: precomputed by the full pipeline unless --recompute.
    for c in MODEL_FEATURES:
        out[c] = docs[c].to_numpy()
    recompute = [r for r in (args.recompute or "").split(",") if r]
    if recompute:
        from pretrain_quality.features import models
        for r in recompute:
            if r not in RECOMPUTABLE:
                raise SystemExit(f"--recompute: unknown '{r}' (choose from {RECOMPUTABLE})")
            t1 = time.time()
            if r == "entities":
                out[ENTITY_FEATURES] = models.score_entities(texts)
            elif r == "textbook":
                out["textbook_quality"] = models.score_textbook(texts)
            elif r == "ai":
                out["ai_generated"] = models.score_ai_detector(texts)
            elif r == "qurater":
                out[QURATER_FEATURES] = models.score_qurater(texts)
            print(f"[extract] recomputed {r} in {time.time() - t1:.1f}s")
    print(f"[extract] model features: "
          f"{'recomputed ' + ','.join(recompute) + '; ' if recompute else ''}"
          f"rest precomputed by the full pipeline")

    ref_path = SAMPLE_DIR / "reference.parquet"
    if ref_path.exists() and not args.limit:
        ref = pd.read_parquet(ref_path).set_index("doc_id").loc[out["doc_id"]]
        ok = []
        for c in CPU_FEATURES:
            a, b = out[c].to_numpy(float), ref[c].to_numpy(float)
            ok.append(np.mean(np.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True)))
        print(f"[extract] check vs. full-pipeline values: {np.mean(ok) * 100:.2f}% of "
              f"doc x feature cells match (worst feature: "
              f"{CPU_FEATURES[int(np.argmin(ok))]} {min(ok) * 100:.2f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[extract] wrote {args.out}  shape={out.shape}")


# ------------------------------------------------------------------ fit

def _r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _design(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    X = df[NO_PC_FEATURES + CAT_FEATURES].copy()
    for c in NO_PC_FEATURES:
        X[c] = X[c].astype(float).fillna(medians[c])
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X


def fit(args) -> None:
    import joblib
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_parquet(args.features)
    medians = json.loads((RESULTS_DIR / "feature_medians.json").read_text())
    X, y = _design(df, medians), df["y"].to_numpy(float)
    print(f"[fit] {len(df):,} docs, {len(NO_PC_FEATURES)} continuous + "
          f"{len(CAT_FEATURES)} categorical features (EBM-no-PC feature set)")
    summary = {"n_docs": int(len(df)), "n_continuous": len(NO_PC_FEATURES),
               "n_categorical": len(CAT_FEATURES)}

    # (a) the full-data model; every sample doc is from its held-out test split
    full_path = paths.REPO_ROOT / "models" / "ebm_no_pc.joblib"
    if full_path.exists():
        full = joblib.load(full_path)
        r2_full = _r2(y, full.predict(X[list(full.feature_names_in_)]))
        summary["full_data_ebm_r2_on_sample"] = r2_full
        print(f"[fit] bundled full-data EBM-no-PC (trained on 1.65M docs), "
              f"R^2 on these held-out docs: {r2_full:.3f}")

    # (b) fit from scratch on the sample
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=args.seed,
                              stratify=df["topic"].to_numpy())
    summary.update(n_train=int(len(tr)), n_test=int(len(te)))

    topic_means = pd.Series(y[tr]).groupby(df["topic"].to_numpy()[tr]).mean()
    p_topic = pd.Series(df["topic"].to_numpy()[te]).map(topic_means).fillna(y[tr].mean())
    summary["topic_only_r2"] = _r2(y[te], p_topic.to_numpy())

    t0 = time.time()
    lasso = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NO_PC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
             CAT_FEATURES)])),
        ("model", LassoCV(cv=5, alphas=np.logspace(-5, 0, 30), max_iter=20000, n_jobs=-1)),
    ]).fit(X.iloc[tr], y[tr])
    summary["lasso_r2"] = float(lasso.score(X.iloc[te], y[te]))
    t_lasso = time.time() - t0

    t0 = time.time()
    # Same hyperparameters as pipeline/scripts/08_refit_ebm_nominal.py.
    ebm = ExplainableBoostingRegressor(
        feature_types=["continuous"] * len(NO_PC_FEATURES) + ["nominal"] * len(CAT_FEATURES),
        feature_names=list(X.columns),
        max_bins=256, interactions=args.interactions, learning_rate=0.01,
        max_rounds=8000, min_samples_leaf=50, outer_bags=args.outer_bags, inner_bags=2,
        random_state=args.seed, n_jobs=-1,
    ).fit(X.iloc[tr], y[tr])
    summary["ebm_r2"] = float(ebm.score(X.iloc[te], y[te]))
    summary["ebm_train_r2"] = float(ebm.score(X.iloc[tr], y[tr]))
    t_ebm = time.time() - t0

    gd = ebm.explain_global().data()
    top = sorted(zip(gd["names"], gd["scores"]), key=lambda kv: -kv[1])[:args.top]
    summary["ebm_top_features"] = [{"feature": n, "importance": float(s)} for n, s in top]

    print(f"[fit] trained on {len(tr):,}, tested on {len(te):,} (stratified by topic)")
    print(f"      topic-only  R^2 = {summary['topic_only_r2']:.3f}")
    print(f"      Lasso       R^2 = {summary['lasso_r2']:.3f}   ({t_lasso:.0f}s)")
    print(f"      EBM-no-PC   R^2 = {summary['ebm_r2']:.3f}   (train {summary['ebm_train_r2']:.3f}, "
          f"{t_ebm:.0f}s)")
    print(f"[fit] top {args.top} EBM terms by mean |contribution|:")
    for i, (n, s) in enumerate(top, 1):
        print(f"      {i:2d}. {n:42s} {s:.5f}")

    RUNS_DIR.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump(ebm, RUNS_DIR / "ebm_no_pc_sample.joblib")
    print(f"[fit] wrote {args.out} and {RUNS_DIR / 'ebm_no_pc_sample.joblib'}")


# ------------------------------------------------------------------ cli

def main() -> None:
    ap = argparse.ArgumentParser(prog="pq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="compute per-document features")
    e.add_argument("--docs", default=str(SAMPLE_DIR / "docs.parquet"))
    e.add_argument("--out", default=str(RUNS_DIR / "features.parquet"))
    e.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0))
                   if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
    e.add_argument("--limit", type=int, default=None, help="only the first N docs")
    e.add_argument("--recompute", default="",
                   help=f"comma list of model features to rerun instead of using the "
                        f"precomputed values: {','.join(RECOMPUTABLE)} "
                        f"(ai/qurater want a GPU; textbook downloads a large fastText model)")
    e.set_defaults(func=extract)

    f = sub.add_parser("fit", help="fit Lasso + EBM-no-PC and report R^2 / top features")
    f.add_argument("--features", default=str(RUNS_DIR / "features.parquet"))
    f.add_argument("--out", default=str(RUNS_DIR / "fit_summary.json"))
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--interactions", type=int, default=20)
    f.add_argument("--outer-bags", type=int, default=4)
    f.add_argument("--top", type=int, default=10)
    f.set_defaults(func=fit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
