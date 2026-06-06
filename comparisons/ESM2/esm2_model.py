"""Pretrained Meta ESM2 (via HuggingFace) fine-tuned for PSI regression on RNA.

Instead of mapping RNA onto ESM2's protein vocabulary, this baseline uses a
dedicated single-nucleotide RNA tokenizer with its own token embedding trained
from scratch; all other ESM2 weights stay pretrained. PSI is exon-specific, so
the baseline (1) crops a window centered on the exon of interest, (2) injects the
exon/intron/flank annotation track as a learned embedding alongside the token
embedding, and (3) pools the encoder output over the exon positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# Dataloader encodings (see src/data/splicedata_dataloader.py).
#   sequence vocab : {PAD:0, A:1, G:2, U:3, C:4, X:5}
#   annotation map : {PAD:0, EXON:1, INTRON:2, FLANK:3}
# Dedicated RNA tokenizer.
RNA_VOCAB = {"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "A": 4, "G": 5, "U": 6, "C": 7, "N": 8}
# dataset sequence token id -> RNA_VOCAB id (PAD/0 is dropped before use).
DATASET_TO_RNA = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}

ANNOTATION_PAD = 0
ANNOTATION_EXON = 1
NUM_ANNOTATION_TOKENS = 4


class ESM2(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        context: int = 512,
        use_pretrained: bool = False,
    ):
        super().__init__()
        self.context = context
        if use_pretrained:
            self.esm = AutoModel.from_pretrained(model_name)
        else:
            self.esm = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        # We feed inputs_embeds (input_ids=None), so ESM's token_dropout branch
        # (which indexes input_ids) must be disabled.
        if getattr(self.esm.embeddings, "token_dropout", False):
            self.esm.embeddings.token_dropout = False

        hidden = self.esm.config.hidden_size
        self.cls_id = RNA_VOCAB["<cls>"]
        self.eos_id = RNA_VOCAB["<eos>"]
        self.pad_id = RNA_VOCAB["<pad>"]

        # RNA token embedding trained from scratch (replaces ESM's protein embedding).
        self.rna_embedding = nn.Embedding(len(RNA_VOCAB), hidden, padding_idx=self.pad_id)
        nn.init.normal_(self.rna_embedding.weight, std=0.02)
        with torch.no_grad():
            self.rna_embedding.weight[self.pad_id].zero_()

        self.annotation_embedding = nn.Embedding(NUM_ANNOTATION_TOKENS, hidden)
        nn.init.zeros_(self.annotation_embedding.weight)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        dataset_to_rna = torch.full((len(DATASET_TO_RNA) + 1,), self.pad_id, dtype=torch.long)
        for dataset_id, rna_id in DATASET_TO_RNA.items():
            dataset_to_rna[dataset_id] = rna_id
        self.register_buffer("dataset_to_rna", dataset_to_rna, persistent=False)

    def _build_inputs(self, sequence: torch.Tensor, annotation: torch.Tensor):
        """Center-crop on the exon, tokenize to RNA ids, align annotation/mask."""
        device = sequence.device
        batch_size, seq_len = sequence.shape
        crop = self.context - 2  # leave room for <cls> and <eos>

        sequence = sequence.long()
        annotation = annotation.long()

        tokens = torch.full((batch_size, self.context), self.pad_id, dtype=torch.long, device=device)
        annotations = torch.zeros((batch_size, self.context), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.context), dtype=torch.long, device=device)

        for i in range(batch_size):
            seq_i = sequence[i]
            ann_i = annotation[i]
            content = (seq_i != ANNOTATION_PAD).nonzero(as_tuple=True)[0]
            exon = (ann_i == ANNOTATION_EXON).nonzero(as_tuple=True)[0]
            if exon.numel() > 0:
                center = int(exon.float().mean().round().item())
            elif content.numel() > 0:
                center = int(content.float().mean().round().item())
            else:
                center = seq_len // 2

            start = max(0, center - crop // 2)
            end = min(seq_len, start + crop)
            start = max(0, end - crop)

            win_seq = seq_i[start:end]
            win_ann = ann_i[start:end]
            keep = win_seq != ANNOTATION_PAD
            win_seq = win_seq[keep]
            win_ann = win_ann[keep]
            n = int(win_seq.numel())

            tokens[i, 0] = self.cls_id
            if n > 0:
                tokens[i, 1 : 1 + n] = self.dataset_to_rna[win_seq]
                annotations[i, 1 : 1 + n] = win_ann
            tokens[i, 1 + n] = self.eos_id
            attention_mask[i, : 2 + n] = 1

        return tokens, annotations, attention_mask

    def forward(
        self,
        sequence: torch.Tensor,
        annotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens, annotations, attention_mask = self._build_inputs(sequence, annotation)

        inputs_embeds = self.rna_embedding(tokens) + self.annotation_embedding(annotations)
        outputs = self.esm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]

        exon_mask = (annotations == ANNOTATION_EXON).type_as(hidden)  # [B, T]
        denom = exon_mask.sum(dim=1, keepdim=True)
        pooled = (hidden * exon_mask.unsqueeze(-1)).sum(dim=1) / denom.clamp(min=1.0)
        # Fall back to [CLS] for the rare sample with no exon positions in view.
        no_exon = denom.squeeze(1) == 0
        if no_exon.any():
            pooled = pooled.clone()
            pooled[no_exon] = hidden[no_exon, 0]

        logits = self.regression_head(pooled)
        return {"logits": logits}
