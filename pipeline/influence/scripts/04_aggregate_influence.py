"""Step 5: aggregate partition scores -> the EBM influence target.

Loads the [1 x n_train] aggregated pairwise scores (all partitions merged),
applies the sign convention (higher = lowers val loss = better quality), and
Gaussian-rank-transforms (EK-FAC magnitudes are uncalibrated in the
non-converged regime: rank ordering is the usable signal).

Outputs, aligned 1:1 with master.parquet train-pool rows / the DiM target:
  data/document_quality_target_influence.npy   (float32, len n_train)
  data/influence_scores.parquet                (shard_idx, doc_idx, raw, quality)

Then refit the EBM with:
  python3 07_train_ebm.py --target data/document_quality_target_influence.npy
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import TRAIN_DOC_INDEX, TRAIN_DOC_OFFSETS, DATA_DIR  # noqa
from influence.task import LMInfluenceTask  # noqa: F401  (kept for symmetry)
from kronfluence.analyzer import Analyzer  # noqa


def gaussian_rank(x: np.ndarray) -> np.ndarray:
    """Rank -> uniform -> inverse-normal CDF (approximately N(0,1) target)."""
    r = rankdata(x, method="average")
    return norm.ppf((r - 0.5) / len(x)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-name", default="doc_attribution")
    ap.add_argument("--scores-name", default="doc_scores")
    ap.add_argument("--output-dir", default=str(DATA_DIR / "influence"))
    ap.add_argument("--sign", type=float, default=-1.0,
                    help="quality = sign * raw_influence. Default -1: kronfluence "
                         "positive score raises the val-loss measurement (hurts), so "
                         "we negate. VERIFY empirically in the Step-1 sanity check.")
    ap.add_argument("--target-out", default=str(DATA_DIR / "document_quality_target_influence.npy"))
    ap.add_argument("--parquet-out", default=str(DATA_DIR / "influence_scores.parquet"))
    args = ap.parse_args()

    # Analyzer only needs output_dir/analysis_name to locate saved scores; load
    # via the module-level loader to avoid constructing a model here.
    from kronfluence.score.pairwise import load_pairwise_scores
    scores_dir = Path(args.output_dir) / args.analysis_name / f"scores_{args.scores_name}"
    blob = load_pairwise_scores(output_dir=scores_dir)
    raw = np.asarray(blob["all_modules"], dtype=np.float64)
    if raw.ndim == 2:                      # [n_query(=1 aggregated), n_train]
        assert raw.shape[0] == 1, f"expected aggregated single query, got {raw.shape}"
        raw = raw[0]
    n_train = len(np.load(TRAIN_DOC_OFFSETS)) - 1
    assert len(raw) == n_train, f"scores len {len(raw)} != n_train {n_train}"
    print(f"loaded {len(raw):,} raw influence scores")

    quality = args.sign * raw
    qrank = gaussian_rank(quality)
    np.save(args.target_out, qrank)
    print(f"wrote {args.target_out}  (gaussian-rank, mean={qrank.mean():.3f} std={qrank.std():.3f})")

    doc_index = np.load(TRAIN_DOC_INDEX)   # (n_train, 2) = [shard_idx, local_idx]
    df = pd.DataFrame({
        "shard_idx": doc_index[:, 0],
        "doc_idx": doc_index[:, 1],
        "raw_influence": raw.astype(np.float32),
        "quality_score": qrank,
    })
    df.to_parquet(args.parquet_out, index=False)
    print(f"wrote {args.parquet_out}  ({len(df):,} rows)")
    # Influence scores are heavy-tailed; report their distribution.
    pct = np.percentile(raw, [0, 1, 50, 99, 100])
    print(f"raw influence pctiles [0,1,50,99,100] = {np.round(pct, 4)}")


if __name__ == "__main__":
    main()
