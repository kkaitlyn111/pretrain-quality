"""Step 3: fit EK-FAC factors ONCE on the trained attribution model.

Covariance (A, S) -> eigendecomposition -> eigenvalue-corrected Lambda, all in
the Kronecker eigenbasis. Written under <output-dir>/<analysis>/factors_<name>/
and loaded read-only by every Step-4 scoring shard.

Usage:
  # end-to-end check (random model, CPU, tiny)
  python3 02_fit_factors.py --random-model --preset 15m --n-fit 32 \
      --per-device-batch-size 4 --cpu --factors-name smoke
  # real fit (after 01_train_attribution_model.py finishes)
  python3 02_fit_factors.py --ckpt data/models/full_pool_100m.pt --n-fit 100000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import (TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, SEQ_LEN, MODELS_DIR,  # noqa
                    DATA_DIR)
from influence.datasets import OneDocDataset  # noqa
from influence.task import LMInfluenceTask  # noqa
from influence.model_io import (build_attribution_model, load_attribution_model,  # noqa
                                count_tracked_linear)
from kronfluence.analyzer import Analyzer, prepare_model  # noqa
from kronfluence.arguments import FactorArguments  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS_DIR / "full_pool_100m.pt"))
    ap.add_argument("--random-model", action="store_true",
                    help="skip checkpoint; random-init model for wiring tests")
    ap.add_argument("--preset", default="100m")
    ap.add_argument("--analysis-name", default="doc_attribution")
    ap.add_argument("--factors-name", default="ekfac_100m")
    ap.add_argument("--output-dir", default=str(DATA_DIR / "influence"))
    ap.add_argument("--n-fit", type=int, default=100_000,
                    help="random subset of train docs used to estimate factors")
    ap.add_argument("--ctx-len", type=int, default=SEQ_LEN)
    ap.add_argument("--per-device-batch-size", type=int, default=8)
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    print(f"device={device}  factors_name={args.factors_name}")

    if args.random_model:
        model, preset, n_layers = build_attribution_model(
            args.preset, attn_impl=args.attn_impl, device="cpu")
        print(f"[random-model] preset={preset} n_layers={n_layers}")
    else:
        model, preset, n_layers = load_attribution_model(
            args.ckpt, attn_impl=args.attn_impl, device="cpu")
        print(f"loaded {args.ckpt}  preset={preset} n_layers={n_layers}")

    task = LMInfluenceTask(n_layers=n_layers)
    model = prepare_model(model=model, task=task)
    n_tracked = count_tracked_linear(model)
    expected = 3 * n_layers
    print(f"tracked MLP linears: {n_tracked} (expected {expected})")
    assert n_tracked == expected, "tracked-module mismatch: check module names"

    analyzer = Analyzer(analysis_name=args.analysis_name, model=model, task=task,
                        output_dir=args.output_dir, cpu=(device == "cpu"),
                        disable_tqdm=False)

    full = OneDocDataset(TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, ctx_len=args.ctx_len)
    rng = np.random.default_rng(args.seed)
    n_fit = min(args.n_fit, len(full))
    idx = sorted(rng.choice(len(full), size=n_fit, replace=False).tolist())
    fit_ds = Subset(full, idx)
    print(f"fitting factors on {n_fit:,} / {len(full):,} docs (ctx={args.ctx_len})")

    # strategy="ekfac"; use_empirical_fisher=False (default) -> TRUE Fisher via sampling
    factor_args = FactorArguments(strategy="ekfac")
    assert factor_args.use_empirical_fisher is False, "must use true (sampled) Fisher"

    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=fit_ds,
        per_device_batch_size=args.per_device_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=args.overwrite,
    )
    print(f"DONE: factors written under {args.output_dir}/{args.analysis_name}/")


if __name__ == "__main__":
    main()
