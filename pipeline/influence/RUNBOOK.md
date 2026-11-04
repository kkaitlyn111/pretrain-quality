# EK-FAC influence target

This pipeline is an alternative to the swarm's difference-in-means target. It scores each
training document with a single-model EK-FAC influence function
([kronfluence](https://github.com/pomonam/kronfluence)) and refits the EBM on that score.

This approach gave a negative result. A 100M-parameter model trained on
the full 1.8B-token pool gives one influence score per document, but these scores predict the
swarm target *y* with held-out R² of only about 0.2. At this model size, influence functions
are not a substitute for randomized swarms.

Install with `pip install -e ".[influence]"`. Below, `data/` means `$PQ_WORK_DIR`. The scripts
are in `pipeline/influence/scripts/` and the SLURM wrappers are
`pipeline/slurm/influence_*.sh`. All scores are aligned row for row with the training-pool rows
of `master.parquet` and with `document_quality_target.npy` (2,063,051 documents).

## Components

- `datasets.py`: `OneDocDataset`, one training document per example in pool order, and
  `QuerySeqDataset`, CommonCrawl validation sequences.
- `task.py`: `LMInfluenceTask`, which uses summed cross-entropy and a sampled (true) Fisher, and
  tracks only the MLP layers.
- `model_io.py`: loads or builds the attribution model with eager attention and without
  `torch.compile`, so that kronfluence's hooks work.

## Steps

1. Attribution model.
   `sbatch pipeline/slurm/influence_train.sh` trains the `100m` preset (101.6M parameters)
   for one epoch over the training pool. The run is resumable and writes
   `data/models/full_pool_100m.pt`.
2. EK-FAC factors. This step needs one 80 GB GPU:
   `sbatch --dependency=afterok:<train-job> pipeline/slurm/influence_fit_factors.sh`.
   The factors are written once to `data/influence/` and loaded read-only by every scoring
   task.
3. Check on a small slice. Before the full sweep, score a single partition to confirm the
   sign convention and that the top influencers look sensible. kronfluence's positive score
   means the document raises validation loss; aggregation flips the sign by default
   (`--sign -1`).
   ```bash
   python pipeline/influence/scripts/03_score_influence.py --n-partitions 4000 --partition-id 0 \
       --scores-name check --query-limit 64 --per-device-train-batch-size 8
   ```
4. Score every document. This step is sharded and resumable; resubmitting the same array
   fills any gaps.
   ```bash
   N=64; sbatch --array=0-$((N-1)) pipeline/slurm/influence_score.sh $N
   ```
5. Aggregate, refit and compare.
   `sbatch pipeline/slurm/influence_finalize.sh` performs three steps:
   - writes `data/document_quality_target_influence.npy`;
   - refits the EBM on that target (outputs tagged `influence`);
   - reports Spearman(influence, *y*) and the two EBMs' R².

## Notes

- Per-sample gradients of the tracked MLP layers dominate memory. On 40 GB GPUs, use
  `--per-device-train-batch-size 4`.
- `damping_factor=None` uses kronfluence's default heuristic of 0.1 × the mean eigenvalue.
