"""Build / load the attribution model for influence computation.

We construct the Llama config directly (rather than via train_lm.build_model)
so we can FORCE `attn_implementation="eager"`: flash-attention-2 and
torch.compile interfere with kronfluence's per-sample-gradient hooks. The
checkpoint is the plain bf16 state_dict written by train_lm.train_and_eval.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def _llama(preset_dims, vocab_size: int, attn_impl: str) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=preset_dims["hidden_size"],
        intermediate_size=preset_dims["intermediate_size"],
        num_hidden_layers=preset_dims["num_hidden_layers"],
        num_attention_heads=preset_dims["num_attention_heads"],
        num_key_value_heads=preset_dims["num_key_value_heads"],
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        use_cache=False,
        rms_norm_eps=1e-5,
        attn_implementation=attn_impl,
    )
    return LlamaForCausalLM(cfg)


def _presets():
    from train_lm import MODEL_PRESETS  # train_lm dir is on sys.path in the scripts
    return MODEL_PRESETS


def build_attribution_model(preset: str = "100m", vocab_size: int = 50304,
                            attn_impl: str = "eager", dtype=torch.float32,
                            device: str = "cpu"):
    """Fresh (randomly initialized) model: used for end-to-end tests without a checkpoint."""
    dims = _presets()[preset]
    model = _llama(dims, vocab_size, attn_impl).to(device=device, dtype=dtype).eval()
    return model, preset, dims["num_hidden_layers"]


def load_attribution_model(ckpt_path, attn_impl: str = "eager",
                           dtype=torch.float32, device: str = "cpu"):
    """Load a trained checkpoint produced by train_lm.train_and_eval.

    Returns (model, preset, n_layers). The state_dict is stored in bf16 with the
    tied embedding present once; LlamaForCausalLM.load_state_dict round-trips it.
    """
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    preset = blob.get("preset", "100m")
    vocab_size = blob.get("vocab_size", 50304)
    dims = _presets()[preset]
    model = _llama(dims, vocab_size, attn_impl)
    sd = {k: v.to(dtype) for k, v in blob["state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Tied lm_head may be absent from / extra in the saved dict depending on the
    # transformers version; everything else must match.
    crit_missing = [m for m in missing if "lm_head" not in m]
    if crit_missing:
        raise RuntimeError(f"checkpoint missing critical params: {crit_missing[:8]}")
    model.tie_weights()
    model.to(device=device, dtype=dtype).eval()
    return model, preset, dims["num_hidden_layers"]


def count_tracked_linear(model) -> int:
    """Count kronfluence TrackedModule wrappers (sanity check after prepare_model)."""
    from kronfluence.module.tracked_module import TrackedModule
    return sum(1 for _ in (m for m in model.modules() if isinstance(m, TrackedModule)))
