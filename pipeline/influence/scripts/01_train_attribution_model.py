"""Train a single reference LM on the FULL train pool (no quality filtering).

Unlike the validation runs (09b), which select exactly 150M unique tokens per
strategy, this trains on ALL ~2.06M docs / ~1.83B tokens of the train pool for
one epoch. It is the "use everything" baseline for the data-selection story.

Defaults: the ~100M-param preset (hidden=640, inter=1728, 14 layers, ~101.6M total),
1 epoch over the full token stream, eval on the fixed 1024-len val-eval set.
torch.compile is OFF by default (clean checkpoints for the downstream EK-FAC influence
pipeline; opt in with --compile). Saves --n-checkpoints evenly-spaced checkpoints
(final -> data/models/full_pool_<preset>.pt, intermediate -> ..._step<N>.pt).

Usage:
  # short test run (a slice of the pool, ~200 steps; checks the code path and throughput)
  python3 01_train_attribution_model.py --smoke
  # full 1-epoch run (compile off, 4 checkpoints)
  python3 01_train_attribution_model.py --preset 100m --wandb-project pq-attribution
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from config import (TRAIN_TOKEN_IDS, VAL_SEQUENCES_EVAL, SEQ_LEN, MODELS_DIR,
                    TOKENS_META)  # noqa
from train_lm import TrainConfig, train_and_eval, prepare_token_stream  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="100m", choices=["15m", "60m", "100m", "120m"])
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="number of passes over the full token stream")
    ap.add_argument("--batch-size", type=int, default=256, help="effective batch (seqs)")
    ap.add_argument("--micro-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile (OFF by default for clean attribution checkpoints)")
    ap.add_argument("--n-checkpoints", type=int, default=4,
                    help="total checkpoints to save (>=3); intermediate saved at ..._step<N>.pt")
    ap.add_argument("--smoke", action="store_true",
                    help="train ~200 steps on a 1/30 slice to verify the path")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    ap.add_argument("--ckpt-every-steps", type=int, default=200,
                    help="write the rolling resume checkpoint every N steps (preemption recovery)")
    ap.add_argument("--no-resume", action="store_true",
                    help="disable the rolling resume checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- load the full train-pool token stream (memmap, uint32) ----
    meta = json.loads(Path(TOKENS_META).read_text())
    n_tokens_total = meta["n_train_tokens"]
    stream = np.memmap(TRAIN_TOKEN_IDS, dtype=np.uint32, mode="r")
    assert len(stream) == n_tokens_total, (len(stream), n_tokens_total)
    print(f"train pool: {meta['n_train_docs']:,} docs, {len(stream):,} tokens")

    val_sequences = np.load(VAL_SEQUENCES_EVAL)
    print(f"val-eval sequences: {val_sequences.shape}")

    rng = np.random.default_rng(args.seed)
    if args.smoke:
        # take a contiguous 1/30 slice (~61M tokens) and cap steps to ~200
        sl = len(stream) // 30
        stream = np.asarray(stream[:sl])
        total_tokens = 200 * args.batch_size * SEQ_LEN
        print(f"[smoke] using {len(stream):,} tokens, {total_tokens:,} train tokens (~200 steps)")
    else:
        # 1 epoch (or args.epochs) over the full stream. prepare_token_stream
        # truncates to total_tokens; >len triggers tiling (multi-epoch).
        total_tokens = int(args.epochs * len(stream))

    t_pack = time.time()
    packed = prepare_token_stream(stream, total_tokens=total_tokens,
                                  seq_len=SEQ_LEN, rng=rng)
    print(f"packed train: {packed.shape} in {time.time()-t_pack:.1f}s")

    ckpt = args.ckpt
    if ckpt is None and not args.smoke:
        ckpt = str(MODELS_DIR / f"full_pool_{args.preset}.pt")

    # Rolling resume checkpoint (single file, overwritten): enables clean
    # continuation after SLURM preemption and requeue. Off for smoke.
    resume_ckpt = None
    if ckpt and not args.no_resume:
        resume_ckpt = (ckpt[:-3] if ckpt.endswith(".pt") else ckpt) + "_resume.pt"

    cfg = TrainConfig(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_len=SEQ_LEN,
        total_tokens=total_tokens,
        lr=args.lr,
        warmup_frac=args.warmup_frac,
        seed=args.seed,
        model_preset=args.preset,
        compile_model=args.compile,
        bf16=True,
        save_checkpoint=ckpt,
        n_checkpoints=args.n_checkpoints,
        resume_ckpt=resume_ckpt,
        ckpt_every_steps=args.ckpt_every_steps,
        log_every=20 if args.smoke else 100,
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.run_name or f"fullpool_{args.preset}_{args.epochs}ep"
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config={**vars(args), "n_train_tokens": len(stream)})

    t0 = time.time()
    result = train_and_eval(packed, val_sequences, cfg, device=device,
                            wandb_run=wandb_run)
    dt = time.time() - t0
    print(f"\n=== DONE in {dt/60:.1f} min ===")
    print(f"val_loss={result['val_loss']:.4f}  steps={result['steps']}  "
          f"params={result['n_params']/1e6:.1f}M  train_time={result['train_time_s']:.0f}s")
    if ckpt:
        print(f"checkpoint: {ckpt}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
