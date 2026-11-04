"""Unit tests for pretrain_quality.features.synth (no corpus needed)."""
from __future__ import annotations

import math

from pretrain_quality.features.synth import SYNTH_FEATURE_COLS, extract_synth_features


def test_all_columns_present():
    out = extract_synth_features("The cat sat on the mat. It was happy because it was warm.")
    assert set(out) == set(SYNTH_FEATURE_COLS)


def test_empty_text_is_nan():
    out = extract_synth_features("")
    assert all(math.isnan(v) for v in out.values())


def test_discourse_and_first_person():
    text = "I like tea. However, my friend prefers coffee because it is strong. Is that odd?"
    out = extract_synth_features(text)
    assert out["discourse_contrastive"] > 0
    assert out["discourse_causal"] > 0
    assert out["question_density"] > 0
    assert 0 < out["first_person_fraction"] <= 1


def test_proper_noun_density_skips_sentence_start():
    # "Paris" and "France" count; sentence-initial "The" does not.
    out = extract_synth_features("The trip went to Paris in France.")
    assert math.isclose(out["proper_noun_density"], 2 / 7 * 1000)
