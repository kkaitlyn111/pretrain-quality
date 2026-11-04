"""Feature manifest shared by the full pipeline and the quickstart."""
from __future__ import annotations

from .heuristics import FEATURE_COLS as HEURISTIC_FEATURES   # 16 Gopher-style
from .synth import SYNTH_FEATURE_COLS as SYNTH_FEATURES      # 18 synthetic-data-inspired

# Pure-CPU, regex/string features: always recomputed from text in the quickstart.
CPU_FEATURES = HEURISTIC_FEATURES + SYNTH_FEATURES

ENTITY_FEATURES = ["entity_density", "entity_diversity", "entity_per_sentence"]  # spaCy
QURATER_FEATURES = ["qr_writing_style", "qr_required_expertise",
                    "qr_facts_and_trivia", "qr_educational_value"]               # QuRater-1.3B (GPU)
# Model-scored features: shipped precomputed in the sample; --recompute to rerun.
MODEL_FEATURES = ENTITY_FEATURES + QURATER_FEATURES + ["textbook_quality", "ai_generated"]

# Benchmark-targeted perplexity-correlation fastText filters; ablated in the headline model.
PC_FEATURES = ["pc_arc_easy", "pc_piqa", "pc_sciq", "pc_lambada"]

# Same order as pipeline/scripts/08_refit_ebm_nominal.py (44 continuous).
NO_PC_FEATURES = CPU_FEATURES + MODEL_FEATURES + ["token_count"]
CAT_FEATURES = ["topic", "format"]  # WebOrganizer topic / format labels (24 each)

TOPIC_NAMES = [
    "Adult", "Art & Design", "Software Dev.", "Crime & Law", "Education & Jobs",
    "Hardware", "Entertainment", "Social Life", "Fashion & Beauty", "Finance & Business",
    "Food & Dining", "Games", "Health", "History", "Home & Hobbies", "Industrial",
    "Literature", "Politics", "Religion", "Science & Tech.", "Software",
    "Sports & Fitness", "Transportation", "Travel",
]
FORMAT_NAMES = [
    "Academic Writing", "Content Listing", "Creative Writing", "Support", "Comment Section",
    "FAQ", "Truncated", "Knowledge Article", "Legal", "Listicle", "News Article",
    "Nonfiction Writing", "About (Org.)", "News (Org.)", "About (Pers.)", "Personal Blog",
    "Product Page", "Q&A", "Spam / Ads", "Structured Data", "Docs.", "Audio Transcript",
    "Tutorial", "User Review",
]
