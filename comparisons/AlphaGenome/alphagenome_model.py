"""AlphaGenome baseline for PSI: frozen pretrained trunk + trainable head.

Uses the PyTorch port (``alphagenome-pytorch``), so the model runs locally on a
GPU with downloaded weights -- no API key or network. Design mirrors
``comparisons/ESM2/esm2_model.py`` where it matters: an exon-centered crop,
exon/intron/flank annotation embeddings added to the sequence representation (from-scratch
mode only), and an MLP regression head pooled over the exon. AlphaGenome's trunk takes
one-hot DNA and we read its 128 bp embeddings (B, L/128, 3072); the exon annotation
selects which 128 bp bins to pool. With ``--use-pretrained``, the trunk is frozen and
only the regression head is trained, matching the original baseline.

From-scratch mode keeps the randomly initialized trunk frozen (450M params cannot be
fit on the comparison sample budget) and trains annotation adapters plus the head,
similar to Evo2's pretrained probing setup.
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

ANNOTATION_PAD = 0
ANNOTATION_EXON = 1
NUM_ANNOTATION_TOKENS = 4

SEQUENCE_LENGTH = 16384
BIN = 128  # encoder downsampling factor (embeddings are at 128 bp resolution).
DNA_EMBED_DIM = 768  # DnaEmbedder output channels.
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
            self.annotation_embedding = None
            self.annotation_embedding_128bp = None
        else:
            self.backbone = AlphaGenome()
            self.annotation_embedding = nn.Embedding(NUM_ANNOTATION_TOKENS, DNA_EMBED_DIM)
            nn.init.zeros_(self.annotation_embedding.weight)
            self.annotation_embedding_128bp = nn.Embedding(NUM_ANNOTATION_TOKENS, EMBED_DIM)
            nn.init.zeros_(self.annotation_embedding_128bp.weight)

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

    def _crop_and_pad(self, seq_row: torch.Tensor, ann_row: torch.Tensor):
        """Center a ``context`` window on the exon and pad it to exact length."""
        length = self.context
        seq_len = seq_row.shape[0]
        content = (seq_row != ANNOTATION_PAD).nonzero(as_tuple=True)[0]
        exon = (ann_row == ANNOTATION_EXON).nonzero(as_tuple=True)[0]
        if exon.numel() > 0:
            center = int(exon.float().mean().round().item())
        elif content.numel() > 0:
            center = int(content.float().mean().round().item())
        else:
            center = seq_len // 2

        start = max(0, center - length // 2)
        end = min(seq_len, start + length)
        start = max(0, end - length)
        win_seq = seq_row[start:end]
        win_ann = ann_row[start:end]

        out_seq = torch.zeros(length, dtype=torch.long, device=seq_row.device)
        out_ann = torch.zeros(length, dtype=torch.long, device=seq_row.device)
        left = (length - win_seq.shape[0]) // 2
        out_seq[left : left + win_seq.shape[0]] = win_seq
        out_ann[left : left + win_ann.shape[0]] = win_ann
        return out_seq, out_ann

    def _build_inputs(self, sequence: torch.Tensor, annotation: torch.Tensor):
        """One-hot DNA (B, L, 4) and a per-bin exon mask (B, L/128)."""
        rows = [self._crop_and_pad(sequence[i], annotation[i]) for i in range(sequence.shape[0])]
        seq_tokens = torch.stack([r[0] for r in rows])
        ann_tokens = torch.stack([r[1] for r in rows])

        channel = self.token_to_channel[seq_tokens]  # (B, L), -1 for PAD/X
        valid = (channel >= 0).unsqueeze(-1).float()
        one_hot = torch.zeros(*channel.shape, 4, device=channel.device)
        one_hot.scatter_(2, channel.clamp(min=0).unsqueeze(-1), valid)

        bins = self.context // BIN
        exon_bins = (ann_tokens == ANNOTATION_EXON).view(-1, bins, BIN).any(dim=2)
        content_bins = (seq_tokens != ANNOTATION_PAD).view(-1, bins, BIN).any(dim=2)
        return one_hot, ann_tokens, exon_bins, content_bins

    def _run_encoder_with_annotation(
        self,
        one_hot: torch.Tensor,
        ann_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder = self.backbone.encoder
        x = one_hot.transpose(1, 2)
        x = encoder.dna_embedder(x)
        x = x + self.annotation_embedding(ann_tokens.long()).transpose(1, 2)

        intermediates: dict[str, torch.Tensor] = {"bin_size_1": x}
        x = encoder.pool(x)
        for i, block in enumerate(encoder.down_blocks):
            if encoder.gradient_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            bin_size = encoder.bin_sizes[i]
            intermediates[f"bin_size_{bin_size}"] = x
            x = encoder.pool(x)
        return x, intermediates

    def _encode_with_annotation(
        self,
        one_hot: torch.Tensor,
        ann_tokens: torch.Tensor,
        organism_index: torch.Tensor,
    ) -> torch.Tensor:
        backbone = self.backbone
        one_hot = backbone.dtype_policy.cast_to_compute(one_hot)
        trunk, _ = self._run_encoder_with_annotation(one_hot, ann_tokens)

        trunk = trunk.transpose(1, 2)
        org_emb = backbone.organism_embed(organism_index).unsqueeze(1)
        trunk = trunk + org_emb
        trunk, _ = backbone.tower(trunk, compute_dtype=backbone.dtype_policy.compute_dtype)

        trunk_ncl = trunk.transpose(1, 2)
        embeddings_128bp = backbone.embedder_128bp(
            trunk_ncl, organism_index, channels_last=False
        )
        return embeddings_128bp.transpose(1, 2).contiguous()

    def _add_annotation_to_128bp_embeddings(
        self,
        embeddings: torch.Tensor,
        ann_tokens: torch.Tensor,
    ) -> torch.Tensor:
        bins = self.context // BIN
        ann_emb = self.annotation_embedding_128bp(ann_tokens.long())
        ann_emb = ann_emb.view(ann_tokens.shape[0], bins, BIN, EMBED_DIM).mean(dim=2)
        return embeddings + ann_emb

    def forward(
        self,
        sequence: torch.Tensor,
        annotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        one_hot, ann_tokens, exon_bins, content_bins = self._build_inputs(
            sequence.long(), annotation.long()
        )
        organism = torch.full(
            (one_hot.shape[0],), self.organism_index, dtype=torch.long, device=one_hot.device
        )
        if self.use_pretrained:
            with torch.no_grad():
                embeddings = self.backbone.encode(one_hot, organism, resolutions=(BIN,))[
                    "embeddings_128bp"
                ].detach()
        else:
            embeddings = self._encode_with_annotation(one_hot, ann_tokens, organism)
            embeddings = self._add_annotation_to_128bp_embeddings(embeddings, ann_tokens)

        def _pool(mask: torch.Tensor) -> torch.Tensor:
            weight = mask.type_as(embeddings).unsqueeze(-1)
            return (embeddings * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

        pooled = _pool(exon_bins)
        no_exon = ~exon_bins.any(dim=1)
        if no_exon.any():
            fallback = _pool(content_bins)
            pooled = torch.where(no_exon.unsqueeze(-1), fallback, pooled)

        return {"logits": self.regression_head(pooled.float())}

    def state_dict(self, *args, **kwargs):
        if self.use_pretrained:
            return {"regression_head": self.regression_head.state_dict()}
        return {
            "annotation_embedding": self.annotation_embedding.state_dict(),
            "annotation_embedding_128bp": self.annotation_embedding_128bp.state_dict(),
            "regression_head": self.regression_head.state_dict(),
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.use_pretrained and "regression_head" in state_dict:
            self.regression_head.load_state_dict(state_dict["regression_head"], strict=strict)
            return
        if not self.use_pretrained and "regression_head" in state_dict:
            self.annotation_embedding.load_state_dict(
                state_dict["annotation_embedding"], strict=strict
            )
            self.annotation_embedding_128bp.load_state_dict(
                state_dict["annotation_embedding_128bp"], strict=strict
            )
            self.regression_head.load_state_dict(state_dict["regression_head"], strict=strict)
            return
        super().load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self
