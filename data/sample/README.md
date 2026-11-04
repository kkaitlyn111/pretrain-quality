# Quickstart sample (10,000 documents)

These are 10,000 documents drawn uniformly at random (seed 0) from the held-out 20% test split
of the EBM fit. That split has 412,611 documents and is stratified by topic, seed 42.
Candidates whose text contains an e-mail address (about 2.4%) were skipped.
Because no sample document was used to train the bundled model `models/ebm_no_pc.joblib`,
scoring the sample with it is an honest held-out check. The sample was built with
`scripts/build_sample.py`, which needs the full pipeline outputs.

## `docs.parquet`

| column | description |
|---|---|
| `doc_id` | `<shard>-<line>` in WebOrganizer/Corpus-200B |
| `text` | full document text |
| `token_count` | GPT-NeoX tokens (from Corpus-200B) |
| `topic`, `format` | WebOrganizer topic / format label ids (0–23; names in `pretrain_quality.features.columns`) |
| `entity_density`, `entity_diversity`, `entity_per_sentence` | spaCy `en_core_web_sm` entity features |
| `qr_writing_style`, `qr_required_expertise`, `qr_facts_and_trivia`, `qr_educational_value` | QuRater-1.3B logits |
| `textbook_quality` | textbook-quality fastText, P(High) + 0.5 * P(Mid) |
| `ai_generated` | RoBERTa OpenAI-detector P(fake) |
| `pc_arc_easy`, `pc_piqa`, `pc_sciq`, `pc_lambada` | perplexity-correlation fastText scores (not used by EBM-no-PC) |
| `y` | per-document quality target from the 593-model swarm (higher = including the doc lowered val loss) |

The model-scored columns are the values the full pipeline computed. `pq extract --recompute …`
reruns them.

## `reference.parquet`

This file holds the full pipeline's values for the 34 CPU heuristic features. `pq extract`
recomputes those features from `text` and checks them against this file. It also holds
`ebm_no_pc_full_pred`, the full-data EBM-no-PC prediction for each document.

## Provenance and license

The text comes from [WebOrganizer/Corpus-200B](https://huggingface.co/datasets/WebOrganizer/Corpus-200B),
which is derived from Common Crawl via the DCLM pool. Use of the text is subject to that
dataset's license and Common Crawl's terms of use, not the MIT license of this repository.
Web text may contain offensive content.
