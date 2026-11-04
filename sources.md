# Third-party data and models

The code in this repository is MIT-licensed (see `LICENSE`). The following bundled or
downloaded artifacts keep their own terms:

| artifact | source | where |
|---|---|---|
| document text, topic/format labels, token counts | [WebOrganizer/Corpus-200B](https://huggingface.co/datasets/WebOrganizer/Corpus-200B) (Common Crawl via DCLM) | `data/sample/docs.parquet` |
| English stopword list | [NLTK](https://www.nltk.org/) stopwords corpus | `src/pretrain_quality/features/stopwords_en.txt` |
| QuRater-1.3B scores | [princeton-nlp/QuRater-1.3B](https://huggingface.co/princeton-nlp/QuRater-1.3B) | precomputed sample columns; downloaded on `--recompute qurater` |
| textbook-quality scores | [kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2](https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2) | precomputed; downloaded on `--recompute textbook` |
| AI-text detector scores | [openai-community/roberta-base-openai-detector](https://huggingface.co/openai-community/roberta-base-openai-detector) | precomputed; downloaded on `--recompute ai` |
| perplexity-correlation scores | [perplexity-correlations](https://huggingface.co/perplexity-correlations) fastText filters | precomputed |
| entity features | [spaCy `en_core_web_sm`](https://spacy.io/models/en) | precomputed; recomputed on `--recompute entities` |
