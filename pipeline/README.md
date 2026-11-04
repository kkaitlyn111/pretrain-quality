# Full pipeline

This is the end-to-end run behind the paper. It assumes a SLURM cluster with A6000/A100-class
GPUs and roughly 1 TB of scratch space. All paths come from environment variables:

```bash
cp config/cluster.env.example config/cluster.env   # edit paths, account, partitions
source config/cluster.env
pip install -e ".[pipeline]"                        # ".[influence]" adds the EK-FAC extension
```

Submit the `pipeline/slurm/*.sh` scripts from the repository root; they source
`config/cluster.env` themselves. Weights & Biases logging is optional. To use it, export
`WANDB_API_KEY` in your shell or run `wandb login`. To turn it off, pass `--no-wandb`.

| directory | contents |
|---|---|
| `features/` | per-shard feature extraction over Corpus-200B |
| `eda/` | filter-score correlations (paper Section 2.2) |
| `factor_model/` | two-factor latent measurement model and selection policies (Sections 3.1, 4.1, 5.1) |
| `scripts/` | swarm target, EBM, downstream validation and figures (Sections 3.2–5.3) |
| `influence/` | EK-FAC influence target, reported as a negative result in Section 6 |
| `slurm/` | SLURM wrappers |

## 1. Corpus and shard sample

Download [WebOrganizer/Corpus-200B](https://huggingface.co/datasets/WebOrganizer/Corpus-200B)
into `$PQ_CORPUS_DIR`. You need the documents, `scores_dclm-fasttext`, `scores_fineweb-edu`,
`domains_*` and `tokens`. Then:

```bash
cd pipeline/features
python 00_subsample_shards.py          # 100 of 9,888 shards, seed 42
python 01_load_annotations.py          # topic, format, DCLM, FineWeb-Edu -> annotations.parquet
cd ../..
python pipeline/scripts/00_split_train_val.py   # hold out 10 shards for LM evaluation
```

## 2. Per-document features

| script | features | compute |
|---|---|---|
| `features/02_extract_heuristics.py` | 16 Gopher-style text statistics | CPU |
| `features/03_extract_synth_features.py` | 18 discourse, self-containedness, structure and readability features | CPU |
| `features/04_extract_entities.py`, `04_extract_entities_batched.py` | 3 spaCy entity features | CPU |
| `features/05_score_textbook_quality.py` | textbook-quality fastText score | CPU |
| `features/08_score_perplexity_corr.py` | 4 perplexity-correlation fastText scores | CPU |
| `features/06_score_ai_detector.py` | RoBERTa AI-text detector | GPU |
| `features/07_score_qurater.py` | QuRater-1.3B, 4 axes | GPU |

```bash
N=$(python -c "import json,os;print(len(json.load(open(os.environ['PQ_FEATURES_DIR']+'/sampled_shards.json'))['shards']))")
sbatch -p "$PQ_CPU_PARTITION" --array=0-$((N-1))%30 pipeline/slurm/features_cpu.sh
sbatch --array=0-$((N-1))%20 pipeline/slurm/features_gpu.sh
sbatch -p "$PQ_CPU_PARTITION" --array=0-$((N-1))%30 pipeline/slurm/pretokenize.sh
python pipeline/scripts/02_consolidate_tokens.py
python pipeline/scripts/03_build_master.py          # -> $PQ_WORK_DIR/master.parquet
```

## 3. Exploratory analysis

```bash
python pipeline/eda/filter_correlations.py          # global vs. within-topic Spearman, Figure 1
```

## 4. Two-factor latent measurement model

The model takes the six transformed filter scores (DCLM, FineWeb-Edu, and the ARC-Easy, PIQA,
SciQ and LAMBADA perplexity-correlation scores) on a 120K-document sample. Its covariates are
17 text heuristics, topic, format and the top 500 URL domains.

```bash
D=pipeline/factor_model
python $D/01_build_model_frame.py && python $D/02_prepare_model_data.py
python $D/03_fit_baselines.py                       # covariate-only baselines
python $D/04_fit_hierarchical_gibbs.py              # hierarchical regression (reference frontier)
python $D/05_fit_one_factor_em.py --iterations 200  # one-factor ablation
python $D/06_fit_two_factor_em.py --iterations 300  # two-factor model (EM)
python $D/07_evaluate_policies.py                   # Table 1
python $D/08_quality_diversity_frontier.py          # Figure 3, right
python $D/09_policy_robustness.py                   # bootstrap confidence intervals
python $D/10_plot_factor_loadings.py                # Figure 3, left
```

Artifacts are written to `$PQ_WORK_DIR/factor_model/`.

## 5. Diagnostic swarm (593 LMs with 15.4M parameters each)

```bash
python pipeline/scripts/04_generate_configs.py --n-type1 200 --n-type2 300 --n-type3 200 --n-type5 50
python pipeline/scripts/05_trim_configs.py          # keep subsets with >= 150M unique tokens
sbatch --array=0-790%35 pipeline/slurm/diagnostic.sh
```

Each run trains `train_lm.py` for one epoch over exactly 150M unique tokens of its subset. The
model has 12 layers, d = 192 and 6 heads, with tied embeddings, bf16, a compiled backbone and
chunked cross-entropy. Each run writes its validation loss to `diagnostic_results/` and its
selection mask to `selection_masks/`.

## 6. Document target and EBM

```bash
python pipeline/scripts/06_build_targets.py         # y_i: difference in means over runs
python pipeline/scripts/07_train_ebm.py             # Lasso, full EBM, EBM without PC scores
python pipeline/scripts/08_refit_ebm_nominal.py     # final EBM-no-PC with nominal topic/format
```

Documents included in fewer than 20 runs get *y* = 0. This applies to 109 of 2,063,051
documents.

## 7. Downstream validation on 30 fresh shards

```bash
python pipeline/scripts/09_download_fresh_shards.py --n 30 --download   # seed 4242
```

Next, run stage-2 feature extraction and pretokenization on the fresh shards, passing
`$PQ_WORK_DIR/fresh_shards.json` as the first argument to the SLURM scripts. Then:

```bash
python pipeline/scripts/10_prepare_validation.py    # fresh_master.parquet + EBM/Lasso scores
MODEL_PRESET=60m EPOCHS=2 OUT_SUFFIX=_60m_2ep sbatch --array=0-9 pipeline/slurm/validation.sh
OUT_SUFFIX=_60m_2ep sbatch --array=0-9 pipeline/slurm/bench.sh   # HellaSwag, ARC-Easy, LAMBADA
```

## 8. Figures

```bash
python pipeline/scripts/13_make_figures.py          # swarm, target, calibration, EBM shapes
```

## EK-FAC influence target

See [`influence/RUNBOOK.md`](influence/RUNBOOK.md).
