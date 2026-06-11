"""Evo 2 baseline for PSI: optional pretrained backbone + trainable head/adapters.

Uses ``evo2_7b_base`` (8K context, bfloat16) — no FP8 or Transformer Engine required.
With ``--use-pretrained``, the Evo2 checkpoint is loaded and the backbone stays frozen
(linear probing). Otherwise the architecture is randomly initialized and trained
end-to-end (requires evo2 + vortex).
Design aligned with ``comparisons/ESM2/esm2_model.py`` where it matters for this task:
an exon-centered window from ``comparison_batch_inputs`` and an MLP regression head.
Evo2 is causal (not bidirectional), so we pool the *last* non-pad token.
See: https://github.com/arcinstitute/evo2
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from vortex import logging as vortex_logging
from vortex.model.attention import CrossAttention, SelfAttention
from vortex.model.model import AttentionBlock

try:
    import transformer_engine  # noqa: F401

    HAS_TE = True
except ImportError:
    HAS_TE = False


def _disable_vortex_activations_log_file() -> None:
    """Stop vortex from writing activations_debug.log to the process cwd."""
    handler = vortex_logging.activations_file_handler
    root = logging.getLogger()
    if handler in root.handlers:
        root.removeHandler(handler)
    handler.close()
    Path("activations_debug.log").unlink(missing_ok=True)


_disable_vortex_activations_log_file()

# Dataloader encodings (see src/data/splicedata_dataloader.py).
#   sequence vocab : {PAD:0, A:1, G:2, U:3, C:4, X:5}
#   annotation map : {PAD:0, EXON:1, INTRON:2, FLANK:3}
# CharLevelTokenizer bytes (DNA; U mapped to T).
DATASET_TO_BYTE = {1: 65, 2: 71, 3: 84, 4: 67, 5: 78}

PAD_INDEX = 0  # dataloader sequence PAD

DEFAULT_MODEL = "evo2_7b_base"


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


def _default_embed_layer(backbone) -> str:
    """Last transformer block (ESM2 uses ``last_hidden_state``)."""
    n_blocks = len(backbone.blocks)
    return f"blocks.{n_blocks - 1}"


def _load_evo2_backbone(model_name: str, use_pretrained: bool):
    """Return (CharLevelTokenizer, StripedHyena backbone)."""
    if use_pretrained:
        from evo2 import Evo2

        wrapper = Evo2(model_name)
        return wrapper.tokenizer, wrapper.model

    import pkgutil

    import yaml
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import CharLevelTokenizer
    from vortex.model.utils import dotdict

    if model_name not in CONFIG_MAP:
        raise ValueError(f"Unknown Evo2 model_name {model_name!r}.")
    config_path = CONFIG_MAP[model_name]
    config = yaml.safe_load(pkgutil.get_data("evo2.models", config_path))
    config = dotdict(config)
    if config.get("use_fp8_input_projections", False) and not HAS_TE:
        if "7b" in model_name:
            warnings.warn(
                "Transformer Engine not installed. "
                "Falling back to bf16 projections (use_fp8_input_projections=False). ",
                stacklevel=2,
            )
            config.use_fp8_input_projections = False
        else:
            raise ImportError(
                f"Model '{model_name}' requires FP8 via Transformer Engine, which is not installed."
            )
    return CharLevelTokenizer(512), StripedHyena(config)


class Evo2ForPSI(nn.Module):
    """Evo 2 encoder with PSI regression head."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        context: int = 8192,
        embed_layer: Optional[str] = None,
        max_length: Optional[int] = None,
        use_pretrained: bool = False,
    ):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("Evo2 requires a CUDA GPU (see ArcInstitute/evo2).")
        if max_length is not None:
            warnings.warn(
                "max_length is deprecated; use context= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            context = max_length

        self.context = context
        self.freeze_backbone = use_pretrained
        self.tokenizer, self.backbone = _load_evo2_backbone(model_name, use_pretrained)
        if not _vortex_flash_attn_available():
            warnings.warn(
                "vortex FlashAttention kernels are unavailable on this GPU; "
                "using PyTorch scaled_dot_product_attention instead (slower but compatible).",
                stacklevel=2,
            )
            _use_pytorch_attention(self.backbone)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        hidden_size = self.backbone.config.hidden_size
        self.embed_layer = embed_layer or _default_embed_layer(self.backbone)

        lookup = torch.zeros(len(DATASET_TO_BYTE) + 1, dtype=torch.long)
        for dataset_id, byte_val in DATASET_TO_BYTE.items():
            lookup[dataset_id] = byte_val
        self.register_buffer("dataset_to_byte", lookup, persistent=False)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.regression_head.to(self._backbone_device())

    def _backbone_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _build_inputs(self, sequence: torch.Tensor) -> torch.Tensor:
        """Center-crop on non-pad content and map RNA tokens to DNA bytes."""
        device = sequence.device
        batch_size, seq_len = sequence.shape
        crop = self.context
        pad_id = self.tokenizer.pad_id

        sequence = sequence.long()

        byte_rows: list[torch.Tensor] = []
        for i in range(batch_size):
            seq_i = sequence[i]
            content = (seq_i != PAD_INDEX).nonzero(as_tuple=True)[0]
            if content.numel() > 0:
                center = int(content.float().mean().round().item())
            else:
                center = seq_len // 2

            start = max(0, center - crop // 2)
            end = min(seq_len, start + crop)
            start = max(0, end - crop)

            win_seq = seq_i[start:end]
            keep = win_seq != PAD_INDEX
            win_seq = win_seq[keep]
            n = min(int(win_seq.numel()), crop)
            if n > 0:
                byte_rows.append(self.dataset_to_byte[win_seq[:n]])
            else:
                byte_rows.append(torch.tensor([ord("N")], dtype=torch.long, device=device))

        max_len = max(row.numel() for row in byte_rows)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        for i, bytes_i in enumerate(byte_rows):
            n = bytes_i.numel()
            input_ids[i, :n] = bytes_i
        return input_ids

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def capture_hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden

        capture = self.backbone.get_submodule(self.embed_layer).register_forward_hook(capture_hook)
        try:
            self.backbone.forward(input_ids)
        finally:
            capture.remove()

        return captured["hidden"]

    def _pool_causal(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Pool the last non-pad token so each position has seen all sequence to its left."""
        batch_size, seq_len, _ = hidden.shape
        idx = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        pad_id = self.tokenizer.pad_id
        content = input_ids != pad_id
        last_content_idx = torch.where(content, idx, torch.zeros_like(idx)).max(dim=1).values
        return hidden[torch.arange(batch_size, device=hidden.device), last_content_idx]

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        input_ids = self._build_inputs(sequence)
        hidden = self._embed(input_ids)
        pooled = self._pool_causal(hidden, input_ids)
        return self.regression_head(pooled.float())

    def state_dict(self, *args, **kwargs):
        if self.freeze_backbone:
            return {"regression_head": self.regression_head.state_dict()}
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.freeze_backbone and "regression_head" in state_dict:
            self.regression_head.load_state_dict(state_dict["regression_head"], strict=strict)
            return
        super().load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
