"""Paths and constants for the latent-factor model of filter scores (paper Sections 3.1, 4.1, 5.1).

Inputs come from the feature stage ($PQ_FEATURES_DIR: annotations.parquet, sampled_shards.json,
features/, scores_pc/, scores_qurater/); all artifacts go to $PQ_WORK_DIR/factor_model.
"""
from __future__ import annotations

from pretrain_quality import paths

DATA_ROOT = paths.FEATURES_DIR
ARTIFACTS_DIR = paths.WORK_DIR / "factor_model"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

SAMPLED_SHARDS_JSON = DATA_ROOT / "sampled_shards.json"
ANNOTATIONS_PARQUET = DATA_ROOT / "annotations.parquet"
FEATURES_DIR = DATA_ROOT / "features"
PC_SCORES_DIR = DATA_ROOT / "scores_pc"
QURATER_DIR = DATA_ROOT / "scores_qurater"

# Stage 1-4: model frame, prepared design, baselines, hierarchical Gibbs regression
MODEL_FRAME_PARQUET = ARTIFACTS_DIR / "model_frame.parquet"
MODEL_FRAME_SUMMARY_JSON = ARTIFACTS_DIR / "model_frame_summary.json"
PREPARED_MODEL_NPZ = ARTIFACTS_DIR / "prepared_model_data.npz"
PREPARED_MODEL_SUMMARY_JSON = ARTIFACTS_DIR / "prepared_model_data_summary.json"
PREPARED_SAMPLE_METADATA_PARQUET = ARTIFACTS_DIR / "prepared_sample_metadata.parquet"
BASELINE_RESULTS_JSON = ARTIFACTS_DIR / "baseline_results.json"
BASELINE_PREDICTIONS_NPZ = ARTIFACTS_DIR / "baseline_predictions.npz"
GIBBS_RESULTS_JSON = ARTIFACTS_DIR / "gibbs_results.json"
GIBBS_DRAWS_NPZ = ARTIFACTS_DIR / "gibbs_draws.npz"

# Stage 5-10: latent-factor EM, selection policies, frontier, robustness
LATENT_EM_RESULTS_JSON = ARTIFACTS_DIR / "one_factor_em_results.json"
LATENT_EM_DRAWS_NPZ = ARTIFACTS_DIR / "one_factor_em_outputs.npz"
FACTOR_EM_RESULTS_JSON = ARTIFACTS_DIR / "two_factor_em_results.json"
FACTOR_EM_OUTPUTS_NPZ = ARTIFACTS_DIR / "two_factor_em_outputs.npz"
LATENT_POLICY_RESULTS_JSON = ARTIFACTS_DIR / "policy_results.json"
LATENT_POLICY_SELECTIONS_PARQUET = ARTIFACTS_DIR / "policy_selections.parquet"
FRONTIER_RESULTS_JSON = ARTIFACTS_DIR / "quality_diversity_frontier.json"
ROBUSTNESS_RESULTS_JSON = ARTIFACTS_DIR / "policy_robustness.json"

PC_MODELS = ["arc_easy", "piqa", "sciq", "lambada", "lambada_es"]
MODEL_SCORE_COLUMNS = [
    "dclm_score",
    "fwedu_rounded",
    "pc_arc_easy",
    "pc_piqa",
    "pc_sciq",
    "pc_lambada",
]
PROBABILITY_SCORE_COLUMNS = [
    "dclm_score",
    "pc_arc_easy",
    "pc_piqa",
    "pc_sciq",
    "pc_lambada",
]
QURATER_AXES = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
]

HEURISTIC_COLS = [
    "char_count",
    "word_count",
    "mean_word_length",
    "frac_alpha",
    "frac_digit",
    "frac_punctuation",
    "frac_uppercase",
    "frac_lines_terminal_punct",
    "frac_lines_bullet",
    "type_token_ratio",
    "stopword_fraction",
    "ngram_rep_2",
    "ngram_rep_3",
    "ngram_rep_4",
    "num_lines",
    "mean_line_length",
]

URL_COLS = ["url", "url_netloc", "url_registered_domain"]
LOG1P_FEATURE_COLS = [
    "token_count",
    "char_count",
    "word_count",
    "num_lines",
    "mean_line_length",
]


def shard_stem(shard_idx: int) -> str:
    return f"CC_shard_{shard_idx:08d}_processed"


def parse_shard_idx_from_stem(stem: str) -> int:
    return int(stem.split("_")[2])
