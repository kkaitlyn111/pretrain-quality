"""Import fasttext with its NumPy-2 incompatibility in `predict` patched out."""
from __future__ import annotations

import fasttext
import fasttext.FastText as _ftmod
import numpy as _np


def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    def _check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        return entry + "\n"
    if isinstance(text, list):
        text = [_check(e) for e in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, _np.asarray(all_probs)
    predictions = self.f.predict(_check(text), k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())
    return labels, _np.asarray(probs)


_ftmod._FastText.predict = _patched_predict

__all__ = ["fasttext"]
