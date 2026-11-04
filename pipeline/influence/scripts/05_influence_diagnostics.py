"""Step 5 diagnostics: the headline numbers for the writeup.

1. Spearman(influence, DiM) globally and on the highest-confidence DiM docs.
   rho>0.5 => same signal; rho~0 => measuring something different (investigate).
2. EBM R^2 comparison: read two ebm_summary.json files (DiM vs influence target)
   produced by 07_train_ebm.py. Beating 0.67 => features were noise-limited.

Usage:
  python3 05_influence_diagnostics.py \
      --influence data/document_quality_target_influence.npy \
      --dim data/document_quality_target.npy \
      [--ebm-dim data/ebm_summary.json --ebm-infl data/ebm_summary_influence.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from config import DATA_DIR  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", default=str(DATA_DIR / "document_quality_target_influence.npy"))
    ap.add_argument("--dim", default=str(DATA_DIR / "document_quality_target.npy"))
    ap.add_argument("--ebm-dim", default=None, help="ebm_summary.json for DiM-target EBM")
    ap.add_argument("--ebm-infl", default=None, help="ebm_summary.json for influence-target EBM")
    args = ap.parse_args()

    infl = np.load(args.influence)
    dim = np.load(args.dim)
    assert len(infl) == len(dim), f"length mismatch {len(infl)} vs {len(dim)}"
    print(f"n_docs = {len(infl):,}")

    # DiM zeros out low-coverage docs; restrict to docs with a real DiM signal.
    nz = dim != 0.0
    rho_all, _ = spearmanr(infl[nz], dim[nz])
    print(f"[1] Spearman(influence, DiM) on {nz.sum():,} nonzero-DiM docs: {rho_all:.4f}")
    # highest-confidence DiM docs (top/bottom 10% by |DiM|)
    absd = np.abs(dim)
    thr = np.quantile(absd[nz], 0.90)
    hc = nz & (absd >= thr)
    rho_hc, _ = spearmanr(infl[hc], dim[hc])
    print(f"    Spearman on top-10% |DiM| ({hc.sum():,} docs): {rho_hc:.4f}")

    if args.ebm_dim and args.ebm_infl:
        d = json.loads(Path(args.ebm_dim).read_text())
        e = json.loads(Path(args.ebm_infl).read_text())
        rd = d.get("r2_ebm") or d.get("ebm_r2")
        re = e.get("r2_ebm") or e.get("ebm_r2")
        print(f"[2] EBM test R^2: DiM target: {rd}   influence target: {re}")
    else:
        print("[2] EBM R^2: run 07_train_ebm.py on each target, pass the two ebm_summary.json")


if __name__ == "__main__":
    main()
