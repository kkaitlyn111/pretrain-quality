"""Step 4: compute pairwise influence scores for ONE data partition.

Each job loads the (read-only) EK-FAC factors + the same query set, computes the
aggregated preconditioned query gradient, and scores its slice of the 2.06M-doc
train set. kronfluence's `data_partitions` + `target_data_partitions` give a
deterministic, resumable shard: a job writes only its partition and reruns skip
finished partitions (`overwrite_output_dir=False`).

`aggregate_query_gradients=True` collapses the val query set to one effective
query -> ONE influence score per train doc (the held-out CC val-loss measurement).
`damping_factor=None` uses the 0.1*mean-eigenvalue heuristic (Grosse et al. §5).

Usage:
  # end-to-end check (random model, CPU, one tiny partition)
  python3 03_score_influence.py --random-model --preset 15m --factors-name smoke \
      --n-partitions 2000 --partition-id 0 --per-device-train-batch-size 4 \
      --query-limit 8 --cpu --scores-name smoke
  # real shard k of N (slurm array)
  python3 03_score_influence.py --n-partitions 64 --partition-id $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import (TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, VAL_SEQUENCES_EVAL,  # noqa
                    SEQ_LEN, MODELS_DIR, DATA_DIR)
from influence.datasets import OneDocDataset, QuerySeqDataset  # noqa
from influence.task import LMInfluenceTask  # noqa
from influence.model_io import (build_attribution_model, load_attribution_model,  # noqa
                                count_tracked_linear)
from kronfluence.analyzer import Analyzer, prepare_model  # noqa
from kronfluence.arguments import ScoreArguments  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS_DIR / "full_pool_100m.pt"))
    ap.add_argument("--random-model", action="store_true")
    ap.add_argument("--preset", default="100m")
    ap.add_argument("--analysis-name", default="doc_attribution")
    ap.add_argument("--factors-name", default="ekfac_100m")
    ap.add_argument("--scores-name", default="doc_scores")
    ap.add_argument("--output-dir", default=str(DATA_DIR / "influence"))
    ap.add_argument("--n-partitions", type=int, default=64)
    ap.add_argument("--partition-id", type=int, required=True)
    ap.add_argument("--ctx-len", type=int, default=SEQ_LEN)
    ap.add_argument("--per-device-query-batch-size", type=int, default=8)
    ap.add_argument("--per-device-train-batch-size", type=int, default=8)
    ap.add_argument("--damping", type=float, default=None,
                    help="damping factor; default None -> 0.1*mean-eigenvalue heuristic")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="cap #query sequences (smoke only)")
    ap.add_argument("--train-limit", type=int, default=None,
                    help="cap #train docs via a leading Subset (smoke only)")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"
    print(f"device={device}  partition {args.partition_id}/{args.n_partitions}")

    if args.random_model:
        model, preset, n_layers = build_attribution_model(
            args.preset, attn_impl=args.attn_impl, device="cpu")
    else:
        model, preset, n_layers = load_attribution_model(
            args.ckpt, attn_impl=args.attn_impl, device="cpu")
    print(f"model preset={preset} n_layers={n_layers}")

    task = LMInfluenceTask(n_layers=n_layers)
    model = prepare_model(model=model, task=task)
    assert count_tracked_linear(model) == 3 * n_layers, "tracked-module mismatch"

    analyzer = Analyzer(analysis_name=args.analysis_name, model=model, task=task,
                        output_dir=args.output_dir, cpu=(device == "cpu"),
                        disable_tqdm=False)

    train_ds = OneDocDataset(TRAIN_TOKEN_IDS, TRAIN_DOC_OFFSETS, ctx_len=args.ctx_len)
    if args.train_limit:
        train_ds = Subset(train_ds, list(range(min(args.train_limit, len(train_ds)))))
    query_ds = QuerySeqDataset(VAL_SEQUENCES_EVAL)
    if args.query_limit:
        query_ds = Subset(query_ds, list(range(min(args.query_limit, len(query_ds)))))
    print(f"train docs: {len(train_ds):,}  query seqs: {len(query_ds):,}")

    score_args = ScoreArguments(
        data_partitions=args.n_partitions,
        damping_factor=args.damping,            # None -> heuristic
        aggregate_query_gradients=True,         # one score per doc
    )

    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        factors_name=args.factors_name,
        query_dataset=query_ds,
        train_dataset=train_ds,
        per_device_query_batch_size=args.per_device_query_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        score_args=score_args,
        target_data_partitions=[args.partition_id],
        overwrite_output_dir=args.overwrite,
    )
    print(f"DONE: wrote partition {args.partition_id} of scores '{args.scores_name}'")


if __name__ == "__main__":
    main()
