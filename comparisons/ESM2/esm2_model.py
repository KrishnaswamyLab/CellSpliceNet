"""Pretrained Meta ESM2 (via HuggingFace) fine-tuned for PSI regression on RNA.

Instead of mapping RNA onto ESM2's protein vocabulary, this baseline uses a
dedicated single-nucleotide RNA tokenizer with its own token embedding trained
from scratch; all other ESM2 weights stay pretrained. The baseline crops a
exon-centered window from ``comparison_batch_inputs`` and reads the [CLS]
representation for regression.
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

PAD_INDEX = 0  # dataloader sequence PAD


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

        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        dataset_to_rna = torch.full((len(DATASET_TO_RNA) + 1,), self.pad_id, dtype=torch.long)
        for dataset_id, rna_id in DATASET_TO_RNA.items():
            dataset_to_rna[dataset_id] = rna_id
        self.register_buffer("dataset_to_rna", dataset_to_rna, persistent=False)

    def _build_inputs(self, sequence: torch.Tensor):
        """Center-crop on non-pad content and tokenize to RNA ids."""
        device = sequence.device
        batch_size, seq_len = sequence.shape
        crop = self.context - 2  # leave room for <cls> and <eos>

        sequence = sequence.long()

        tokens = torch.full((batch_size, self.context), self.pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.context), dtype=torch.long, device=device)

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
            n = int(win_seq.numel())

            tokens[i, 0] = self.cls_id
            if n > 0:
                tokens[i, 1 : 1 + n] = self.dataset_to_rna[win_seq]
            tokens[i, 1 + n] = self.eos_id
            attention_mask[i, : 2 + n] = 1

        return tokens, attention_mask

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, attention_mask = self._build_inputs(sequence)

        inputs_embeds = self.rna_embedding(tokens)
        outputs = self.esm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]

        logits = self.regression_head(hidden[:, 0])
        return {"logits": logits}
