"""Pretrained Meta ESM2 (via HuggingFace) fine-tuned for PSI regression on RNA."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ESM2 is protein-trained; map RNA nucleotides to amino-acid tokens ESM accepts.
RNA_TO_AA = {
    0: "X",
    1: "A",
    2: "G",
    3: "S",  # U -> serine (common RNA-to-protein mapping for ESM)
    4: "C",
    5: "X",
}


def rna_tensor_to_strings(sequence: torch.Tensor, max_len: int) -> list[str]:
    out: list[str] = []
    for row in sequence.detach().cpu().tolist():
        chars = [RNA_TO_AA.get(int(tok), "X") for tok in row if int(tok) != 0]
        out.append("".join(chars[:max_len]))
    return out


class ESM2(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        max_length: int = 1024,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name)
        hidden = self.esm.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        annotation: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        del annotation  # ESM2 uses sequence only (pretrained protein tokenizer)
        strings = rna_tensor_to_strings(sequence, self.max_length - 2)
        tokens = self.tokenizer(
            strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        tokens = {k: v.to(sequence.device) for k, v in tokens.items()}
        outputs = self.esm(**tokens)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.regression_head(pooled)
        return {"logits": logits}
