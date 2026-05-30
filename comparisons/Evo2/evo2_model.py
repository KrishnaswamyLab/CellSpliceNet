"""Evo 2 linear-probe baseline: frozen pretrained backbone + trainable regression head.

Uses evo2_7b_base (8K context, no Transformer Engine / FP8 requirement).
See: https://github.com/arcinstitute/evo2
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from evo2 import Evo2
from vortex import logging as vortex_logging
from vortex.model.attention import CrossAttention, SelfAttention
from vortex.model.model import AttentionBlock


def _disable_vortex_activations_log_file() -> None:
    """Stop vortex from writing activations_debug.log to the process cwd."""
    handler = vortex_logging.activations_file_handler
    root = logging.getLogger()
    if handler in root.handlers:
        root.removeHandler(handler)
    handler.close()
    Path("activations_debug.log").unlink(missing_ok=True)


_disable_vortex_activations_log_file()

# CellSpliceNet RNA vocab -> ASCII bytes for CharLevelTokenizer (DNA: U->T).
RNA_TO_BYTE = {
    0: None,
    1: 65,   # A
    2: 71,   # G
    3: 84,   # U -> T
    4: 67,   # C
    5: 78,   # X -> N
}

DEFAULT_MODEL = "evo2_7b_base"
DEFAULT_MAX_LENGTH = 4096
# Mid-depth layer; Evo 2 paper notes intermediate embeddings work well.
DEFAULT_EMBED_LAYER = "blocks.16"
HIDDEN_SIZE = 4096  # evo2_7b_base config hidden_size


def _vortex_flash_attn_available() -> bool:
    """Probe whether vortex FlashAttention CUDA kernels run on the current GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        from vortex.ops import local_flash_attn_qkvpacked_func
    except ImportError:
        return False
    device = torch.device("cuda")
    # evo2_7b: 32 heads, head_dim=128
    qkv = torch.randn(1, 32, 3, 32, 128, device=device, dtype=torch.bfloat16)
    try:
        local_flash_attn_qkvpacked_func(qkv, 0.0, causal=True)
        return True
    except RuntimeError:
        return False


def _use_pytorch_attention(backbone) -> None:
    """Swap vortex FlashAttention modules for PyTorch SDPA (portable across GPU archs)."""
    backbone.config.use_flash_attn = False
    for block in backbone.blocks:
        if not isinstance(block, AttentionBlock):
            continue
        mha = block.inner_mha_cls
        if not mha.use_flash_attn:
            continue
        dropout = mha.inner_attn.drop.p if hasattr(mha.inner_attn, "drop") else 0.0
        mha.use_flash_attn = False
        mha.inner_attn = SelfAttention(causal=mha.causal, attention_dropout=dropout)
        mha.inner_cross_attn = CrossAttention(causal=mha.causal, attention_dropout=dropout)


class Evo2ForPSI(nn.Module):
    """Frozen Evo 2 encoder with a linear PSI head (linear probing)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        embed_layer: str = DEFAULT_EMBED_LAYER,
    ):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("Evo2 requires a CUDA GPU (see ArcInstitute/evo2).")

        self.max_length = max_length
        self.embed_layer = embed_layer
        self._evo2 = Evo2(model_name)
        self.tokenizer = self._evo2.tokenizer
        self.backbone = self._evo2.model
        if not _vortex_flash_attn_available():
            warnings.warn(
                "vortex FlashAttention kernels are unavailable on this GPU; "
                "using PyTorch scaled_dot_product_attention instead (slower but compatible).",
                stacklevel=2,
            )
            _use_pytorch_attention(self.backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.head = nn.Linear(HIDDEN_SIZE, 1)
        self.head.to(self._backbone_device())

    def _backbone_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _encode(self, sequence: torch.Tensor) -> torch.Tensor:
        rows: list[list[int]] = []
        max_len = 0
        for row in sequence.detach().cpu().tolist():
            toks: list[int] = []
            for tok in row:
                byte_val = RNA_TO_BYTE.get(int(tok))
                if byte_val is None:
                    continue
                toks.append(byte_val)
                if len(toks) >= self.max_length:
                    break
            rows.append(toks)
            max_len = max(max_len, len(toks))

        if max_len == 0:
            max_len = 1

        pad_id = self.tokenizer.pad_id
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        for i, toks in enumerate(rows):
            if toks:
                input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        return input_ids

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden

        handle = self.backbone.get_submodule(self.embed_layer).register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.backbone.forward(input_ids)
        finally:
            handle.remove()

        return captured["hidden"]

    def forward(
        self,
        sequence: torch.Tensor,
        annotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del annotation
        input_ids = self._encode(sequence).to(self._backbone_device())
        hidden = self._embed(input_ids)
        pooled = hidden.mean(dim=1)
        return self.head(pooled.float())

    def state_dict(self, *args, **kwargs):
        return self.head.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.head.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self
