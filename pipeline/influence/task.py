"""kronfluence Task for the Llama attribution model.

Adapted from `kronfluence/examples/openwebtext/task.py`. Correctness notes:

- Summed token cross-entropy (`reduction="sum"`), required for the
  Fisher = Gauss-Newton equivalence that EK-FAC relies on.
- `sample=True` draws targets from the model's own predictive distribution
  (true Fisher), NOT the training labels (which would give the worse empirical
  Fisher). This is the single most common source of silently-wrong EK-FAC
  factors, so it is implemented explicitly here.
- Only the MLP / feed-forward linear layers are tracked (paper default;
  `model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}` for HF Llama).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from kronfluence.task import Task


class LMInfluenceTask(Task):
    def __init__(self, n_layers: int):
        self.n_layers = int(n_layers)

    def _shift_logits_labels(self, batch, model):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        return logits, labels

    def compute_train_loss(self, batch, model, sample: bool = False) -> torch.Tensor:
        logits, labels = self._shift_logits_labels(batch, model)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).flatten()
            sampled[labels == -100] = -100        # keep padded positions ignored
        return F.cross_entropy(logits, sampled, reduction="sum", ignore_index=-100)

    def compute_measurement(self, batch, model) -> torch.Tensor:
        logits, labels = self._shift_logits_labels(batch, model)
        return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

    def get_influence_tracked_modules(self) -> List[str]:
        mods: List[str] = []
        for i in range(self.n_layers):
            mods.append(f"model.layers.{i}.mlp.gate_proj")
            mods.append(f"model.layers.{i}.mlp.up_proj")
            mods.append(f"model.layers.{i}.mlp.down_proj")
        return mods

    def get_attention_mask(self, batch):
        return batch["attention_mask"]
