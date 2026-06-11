from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

RNA_MAP = {
    0: "N",
    1: "A",
    2: "G",
    3: "T",  # RNA U -> DNA T (SpliceBERT vocab)
    4: "C",
    5: "N",
}


def sequence_to_list(sequence: torch.Tensor) -> List[str]:
    sequence_list = sequence.cpu().numpy().tolist()
    return [" ".join(RNA_MAP.get(int(num), "N") for num in row if int(num) != 0) for row in sequence_list]


def default_pretrained_path(data_tag: str) -> Path:
    root = Path(__file__).resolve().parent / "pretrained" / "models"
    if data_tag.lower() in ("gtex", "human"):
        return root / "SpliceBERT-human.510nt"
    return root / "SpliceBERT.1024nt"


def default_max_seq_len(data_tag: str) -> int:
    if data_tag.lower() in ("gtex", "human"):
        return 510
    return 1024


class SpliceBert(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        model_path: str | Path | None = None,
        data_tag: str = "replicate",
        max_seq_len: int | None = None,
        use_pretrained: bool = False,
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len if max_seq_len is not None else default_max_seq_len(data_tag)
        model_path = Path(model_path) if model_path is not None else default_pretrained_path(data_tag)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if use_pretrained:
            self.bertmodel = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                problem_type="regression",
                num_labels=1,
            )
        else:
            config = AutoConfig.from_pretrained(str(model_path))
            config.num_labels = 1
            config.problem_type = "regression"
            self.bertmodel = AutoModelForSequenceClassification.from_config(config)
        # Keep first 3 encoder layers (original comparison design).
        self.bertmodel.bert.encoder.layer = self.bertmodel.bert.encoder.layer[:3]
        self.bertmodel.to(self.device)

    def forward(self, sequence: torch.Tensor):
        sequence_list = sequence_to_list(sequence)
        tokens = self.tokenizer(
            sequence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        tokens = tokens.to(self.device)

        outputs = self.bertmodel(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        return outputs.logits
