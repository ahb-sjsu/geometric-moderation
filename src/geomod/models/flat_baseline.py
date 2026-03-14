# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Flat baseline model: DeBERTa + linear classification head.

No geometry — serves as the ablation baseline (Config A) to measure
the contribution of hyperbolic embeddings and geometric attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from geomod.policy.taxonomy import PolicyNode, default_taxonomy


class FlatModerationModel(nn.Module):
    """Flat (non-geometric) content moderation model.

    Architecture: pretrained encoder → [CLS] → linear → logits

    Returns the same dict format as GeometricModerationModel for
    uniform trainer interface, but without severity or embedding fields.

    Parameters
    ----------
    encoder_name : str
        HuggingFace model name.
    taxonomy : PolicyNode | None
        Policy taxonomy (used only to determine number of output classes).
    dropout : float
        Dropout before the classification head.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        taxonomy: PolicyNode | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = config.hidden_size

        if taxonomy is None:
            taxonomy = default_taxonomy()
        num_classes = len(taxonomy.all_nodes())

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode → classify.

        Returns
        -------
        dict with key:
            logits : (batch, num_classes)
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = encoder_output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return {"logits": logits}
