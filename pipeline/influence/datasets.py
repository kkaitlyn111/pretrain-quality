"""Datasets for EK-FAC influence scoring.

Two dataset shapes feed kronfluence:

- `OneDocDataset`: ONE training document per example, in train-pool doc order.
  Even though training packed docs into 1024-token sequences, per-document
  attribution requires one doc per example: we truncate long docs to `ctx_len`
  and pad short ones (attention_mask + labels=-100 on pad so padded positions
  contribute no loss/gradient). Example index `i` aligns 1:1 with
  `master.parquet` train-pool row `i` and `document_quality_target.npy[i]`
  (all built in train-shard order), so the influence vector needs no join.

- `QuerySeqDataset`: the held-out CC val sequences (the measurement set).
  Each row is a pre-packed (ctx+1)-token sequence; the Task shifts internally.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class OneDocDataset(Dataset):
    def __init__(self, token_ids_path, doc_offsets_path, ctx_len: int = 1024,
                 pad_id: int = 0):
        self.offsets = np.load(doc_offsets_path)                 # (n_docs+1,) int64
        self.tokens = np.memmap(token_ids_path, dtype=np.uint32, mode="r")
        self.ctx_len = int(ctx_len)
        self.pad_id = int(pad_id)
        self.n_docs = len(self.offsets) - 1

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        toks = np.asarray(self.tokens[a:b][: self.ctx_len], dtype=np.int64)
        L = len(toks)
        if L == 0:                       # empty doc guard (shouldn't happen post-filter)
            toks = np.array([self.pad_id], dtype=np.int64)
            L = 1
        input_ids = np.full(self.ctx_len, self.pad_id, dtype=np.int64)
        input_ids[:L] = toks
        attn = np.zeros(self.ctx_len, dtype=np.int64)
        attn[:L] = 1
        labels = np.full(self.ctx_len, -100, dtype=np.int64)
        labels[:L] = toks
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn),
            "labels": torch.from_numpy(labels),
        }


class QuerySeqDataset(Dataset):
    """Pre-packed (ctx+1)-token val sequences. The Task shifts input/labels
    internally, so we hand it the full row as both input_ids and labels with an
    all-ones attention mask (these sequences are densely packed, no padding)."""
    def __init__(self, sequences_path):
        arr = np.load(sequences_path)                            # (M, ctx+1) uint32
        self.seqs = np.ascontiguousarray(arr.astype(np.int64))
        self.n, self.width = self.seqs.shape

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        seq = self.seqs[i]
        return {
            "input_ids": torch.from_numpy(seq.copy()),
            "attention_mask": torch.ones(self.width, dtype=torch.long),
            "labels": torch.from_numpy(seq.copy()),
        }
