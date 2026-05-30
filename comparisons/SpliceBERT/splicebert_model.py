from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        vocab_size_annotation: int = 4,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len if max_seq_len is not None else default_max_seq_len(data_tag)
        self.vocab_size_annotation = vocab_size_annotation
        model_path = Path(model_path) if model_path is not None else default_pretrained_path(data_tag)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.bertmodel = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            problem_type="regression",
            num_labels=1,
        )
        # Keep first 3 encoder layers (original comparison design).
        self.bertmodel.bert.encoder.layer = self.bertmodel.bert.encoder.layer[:3]
        self.bertmodel.to(self.device)

        embed_dim = self.bertmodel.bert.embeddings.word_embeddings.embedding_dim
        self.embed_annotation = nn.Embedding(self.vocab_size_annotation, embed_dim)
        self.embed_annotation.to(self.device)

    def forward(self, sequence: torch.Tensor, annotation: torch.Tensor | None = None):
        sequence_list = sequence_to_list(sequence)
        tokens = self.tokenizer(
            sequence_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        tokens = tokens.to(self.device)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        if annotation is not None:
            annotation = annotation.long()
            annotation = annotation[:, : input_ids.shape[-1] - 2]
            annotation = torch.nn.functional.pad(annotation, (1, 1), "constant", 0)
            annotation = annotation.to(self.device)

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if hasattr(self.bertmodel.bert.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.bertmodel.bert.embeddings.token_type_ids[:, :seq_length]
            token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

        embedding_sequence = self.bertmodel.bert.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        if annotation is not None:
            embedding_joint = embedding_sequence + self.embed_annotation(annotation)
        else:
            embedding_joint = embedding_sequence

        extended_attention_mask = self.bertmodel.bert.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.bertmodel.bert.get_head_mask(None, self.bertmodel.bert.config.num_hidden_layers)

        encoder_outputs = self.bertmodel.bert.encoder(
            embedding_joint,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_output = encoder_outputs[0]
        pooled = self.bertmodel.bert.pooler(sequence_output)
        return self.bertmodel.classifier(pooled)
