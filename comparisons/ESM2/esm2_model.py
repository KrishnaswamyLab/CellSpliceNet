"""Pretrained Meta ESM2 (via HuggingFace) fine-tuned for PSI regression on RNA.

ESM2 is protein-trained, so RNA nucleotides are mapped to amino-acid tokens. The
splicing target (PSI) is exon-specific, so this baseline (1) crops a window
centered on the exon of interest, (2) injects the exon/intron/flank annotation
track as a learned embedding alongside the token embedding, and (3) pools the
encoder output over the exon positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Dataloader encodings (see src/data/splicedata_dataloader.py).
#   sequence vocab : {PAD:0, A:1, G:2, U:3, C:4, X:5}
#   annotation map : {PAD:0, EXON:1, INTRON:2, FLANK:3}
RNA_TO_AA = {0: "X", 1: "A", 2: "G", 3: "S", 4: "C", 5: "X"}  # U -> serine (common ESM mapping)
ANNOTATION_PAD = 0
ANNOTATION_EXON = 1
NUM_ANNOTATION_TOKENS = 4


class ESM2(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        context: int = 512,
    ):
        super().__init__()
        self.context = context
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)
        self._token_dropout_scale = 1.0
        if getattr(self.esm.embeddings, "token_dropout", False):
            self._token_dropout_scale = 0.3
            self.esm.embeddings.token_dropout = False

        hidden = self.esm.config.hidden_size
        self.annotation_embedding = nn.Embedding(NUM_ANNOTATION_TOKENS, hidden)
        nn.init.zeros_(self.annotation_embedding.weight)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        rna_to_aa_id = torch.full((len(RNA_TO_AA),), self.tokenizer.pad_token_id, dtype=torch.long)
        for rna_tok, aa in RNA_TO_AA.items():
            rna_to_aa_id[rna_tok] = self.tokenizer.convert_tokens_to_ids(aa)
        self.register_buffer("rna_to_aa_id", rna_to_aa_id, persistent=False)

    def _build_inputs(self, sequence: torch.Tensor, annotation: torch.Tensor):
        """Center-crop on the exon, retokenize to ESM ids, align annotation/mask."""
        device = sequence.device
        batch_size, seq_len = sequence.shape
        crop = self.context - 2  # leave room for [CLS] and [EOS]
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        sequence = sequence.long()
        annotation = annotation.long()

        tokens = torch.full((batch_size, self.context), pad_id, dtype=torch.long, device=device)
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

            tokens[i, 0] = cls_id
            if n > 0:
                tokens[i, 1 : 1 + n] = self.rna_to_aa_id[win_seq]
                annotations[i, 1 : 1 + n] = win_ann
            tokens[i, 1 + n] = eos_id
            attention_mask[i, : 2 + n] = 1

        return tokens, annotations, attention_mask

    def forward(
        self,
        sequence: torch.Tensor,
        annotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens, annotations, attention_mask = self._build_inputs(sequence, annotation)

        inputs_embeds = self.esm.embeddings.word_embeddings(tokens) * self._token_dropout_scale
        inputs_embeds = inputs_embeds + self.annotation_embedding(annotations)
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
