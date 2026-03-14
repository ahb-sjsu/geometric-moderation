# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tokenization and collation for the moderation training pipeline.

Wraps a HuggingFace tokenizer and provides a DataLoader-compatible
collate function that returns padded input_ids + attention_mask + labels.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer


class ModerationTokenizer:
    """Tokenizer wrapper for content moderation inputs.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (must match the encoder).
    max_length : int
        Maximum token length. 256 covers >95% of Civil Comments
        and keeps the attention bias matrix manageable.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts.

        Returns dict with 'input_ids' and 'attention_mask', both (batch, seq).
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: ModerationTokenizer,
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Takes a list of dataset items (from CivilCommentsDataset) and returns
    a batched dict with tokenized inputs and stacked labels.

    Parameters
    ----------
    batch : list of dicts
        Each dict has keys: text, taxonomy_label, taxonomy_labels_multi,
        severity_target.
    tokenizer : ModerationTokenizer
        Tokenizer instance.

    Returns
    -------
    dict with keys:
        input_ids : (batch, seq) token IDs
        attention_mask : (batch, seq)
        taxonomy_label : (batch,) single-label indices
        taxonomy_labels_multi : (batch, num_nodes) multi-hot
        severity_target : (batch,) severity floats
    """
    texts = [item["text"] for item in batch]
    encoded = tokenizer(texts)

    taxonomy_labels = torch.tensor(
        [item["taxonomy_label"] for item in batch], dtype=torch.long
    )
    taxonomy_labels_multi = torch.stack(
        [item["taxonomy_labels_multi"] for item in batch]
    )
    severity_targets = torch.tensor(
        [item["severity_target"] for item in batch], dtype=torch.float
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "taxonomy_label": taxonomy_labels,
        "taxonomy_labels_multi": taxonomy_labels_multi,
        "severity_target": severity_targets,
    }


def make_collate_fn(tokenizer: ModerationTokenizer):
    """Return a collate function bound to a tokenizer (for DataLoader)."""
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_fn(batch, tokenizer)
    return _collate
