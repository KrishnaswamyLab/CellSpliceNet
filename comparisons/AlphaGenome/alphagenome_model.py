"""AlphaGenome baseline for PSI: frozen pretrained trunk + trainable head.

Uses the PyTorch port (``alphagenome-pytorch``), so the model runs locally on a
GPU with downloaded weights -- no API key or network. Design mirrors
``comparisons/ESM2/esm2_model.py`` where it matters: an exon-centered crop from
``comparison_batch_inputs`` (annotation is used only there to center the crop, not
passed to this model), mean-pooling over non-pad 128 bp bins, and an MLP regression head.
AlphaGenome's trunk takes one-hot DNA and we read its 128 bp embeddings
(B, L/128, 3072). With ``--use-pretrained``, the trunk is frozen and only the
regression head is trained, matching the original baseline.

From-scratch mode keeps the randomly initialized trunk frozen (450M params cannot be
fit on the comparison sample budget) and trains only the head, similar to linear
probing on random features.
See: https://github.com/genomicsxai/alphagenome-pytorch
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from alphagenome_pytorch import AlphaGenome

# Dataloader encodings (see src/data/splicedata_dataloader.py).
#   sequence vocab : {PAD:0, A:1, G:2, U:3, C:4, X:5}
#   annotation map : {PAD:0, EXON:1, INTRON:2, FLANK:3}
# AlphaGenome one-hot channels are A=0, C=1, G=2, T=3 (U mapped to T);
# PAD/X map to -1 -> all-zero one-hot vector.
DATASET_TO_CHANNEL = {0: -1, 1: 0, 2: 2, 3: 3, 4: 1, 5: -1}

PAD_INDEX = 0  # dataloader sequence PAD

SEQUENCE_LENGTH = 16384
BIN = 128  # encoder downsampling factor (embeddings are at 128 bp resolution).
EMBED_DIM = 3072  # embeddings_128bp channel dimension.


class AlphaGenomeForPSI(nn.Module):
    """Frozen AlphaGenome trunk (128 bp embeddings) + trainable PSI head."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        context: int = SEQUENCE_LENGTH,
        organism_index: int = 0,
        hidden: int = EMBED_DIM,
        use_pretrained: bool = False,
    ):
        super().__init__()
        if context % BIN != 0:
            raise ValueError(f"context must be a multiple of {BIN}, got {context}.")
        self.context = context
        self.organism_index = organism_index
        self.use_pretrained = use_pretrained

        if use_pretrained:
            try:
                self.backbone = AlphaGenome.from_pretrained("model_all_folds.safetensors")
            except Exception as e:
                print(f"Error loading AlphaGenome pretrained weights: {e}")
                self.backbone = AlphaGenome()
        else:
            self.backbone = AlphaGenome()

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.regression_head = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        channel = torch.full((len(DATASET_TO_CHANNEL),), -1, dtype=torch.long)
        for token, ch in DATASET_TO_CHANNEL.items():
            channel[token] = ch
        self.register_buffer("token_to_channel", channel, persistent=False)

    def _pad_or_trim(self, seq_row: torch.Tensor) -> torch.Tensor:
        """Pad or trim a row to exactly ``self.context`` (centered padding)."""
        length = self.context
        seq_len = seq_row.shape[0]
        if seq_len >= length:
            start = (seq_len - length) // 2
            return seq_row[start : start + length]
        out = torch.zeros(length, dtype=seq_row.dtype, device=seq_row.device)
        left = (length - seq_len) // 2
        out[left : left + seq_len] = seq_row
        return out

    def _build_inputs(self, sequence: torch.Tensor):
        """One-hot DNA (B, L, 4) and per-bin content mask (B, L/128)."""
        seq_tokens = torch.stack([self._pad_or_trim(sequence[i]) for i in range(sequence.shape[0])])

        channel = self.token_to_channel[seq_tokens]  # (B, L), -1 for PAD/X
        valid = (channel >= 0).unsqueeze(-1).float()
        one_hot = torch.zeros(*channel.shape, 4, device=channel.device)
        one_hot.scatter_(2, channel.clamp(min=0).unsqueeze(-1), valid)

        bins = self.context // BIN
        content_bins = (seq_tokens != PAD_INDEX).reshape(seq_tokens.shape[0], bins, BIN).any(dim=2)
        return one_hot, content_bins

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        one_hot, content_bins = self._build_inputs(sequence)
        organism = torch.full(
            (one_hot.shape[0],), self.organism_index, dtype=torch.long, device=one_hot.device
        )
        with torch.no_grad():
            embeddings = self.backbone.encode(one_hot, organism, resolutions=(BIN,))[
                "embeddings_128bp"
            ].detach()

        weight = content_bins.type_as(embeddings).unsqueeze(-1)
        pooled = (embeddings * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

        return {"logits": self.regression_head(pooled.float())}

    def state_dict(self, *args, **kwargs):
        return {"regression_head": self.regression_head.state_dict()}

    def load_state_dict(self, state_dict, strict: bool = True):
        if "regression_head" in state_dict:
            self.regression_head.load_state_dict(state_dict["regression_head"], strict=strict)
            return
        super().load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self
