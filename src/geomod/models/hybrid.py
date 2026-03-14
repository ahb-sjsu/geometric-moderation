# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Hybrid content moderation model.

Combines the best of both approaches:
1. Linear classification head (768→41) for maximum accuracy
2. Parallel Poincaré branch for severity calibration and hierarchy
3. Geometric attention hooks for structured encoder representations

The linear head provides the primary classification decision.
The Poincaré branch is an auxiliary task that teaches the encoder
to organize representations geometrically, while providing severity
scores and interpretable policy distances that the linear head cannot.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from geomod.models.classifier import HyperbolicClassifier
from geomod.policy.taxonomy import PolicyNode, default_taxonomy


class HybridModerationModel(nn.Module):
    """Hybrid linear + geometric content moderation model.

    Parameters
    ----------
    encoder_name : str
        HuggingFace model name.
    taxonomy : PolicyNode
        Policy taxonomy tree.
    hyp_dim : int
        Hyperbolic embedding dimension.
    c : float
        Curvature.
    use_geometric_attention : bool
        Whether to install hyperbolic attention bias hooks.
    temperature : float
        Temperature for distance-based logits in the Poincaré branch.
    dropout : float
        Dropout before the linear classification head.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        taxonomy: PolicyNode | None = None,
        hyp_dim: int = 32,
        c: float = 1.0,
        use_geometric_attention: bool = True,
        temperature: float = 1.0,
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

        # Primary head: linear classification (full 768-dim expressivity)
        self.dropout = nn.Dropout(dropout)
        self.linear_head = nn.Linear(encoder_dim, num_classes)

        # Auxiliary head: Poincaré branch (severity + hierarchy)
        self.poincare_head = HyperbolicClassifier(
            encoder_dim=encoder_dim,
            taxonomy=taxonomy,
            hyp_dim=hyp_dim,
            c=c,
            temperature=temperature,
        )

        # Geometric attention hooks (optional)
        self.geo_wrapper = None
        self.geo_bias = None
        if use_geometric_attention:
            from geomod.models.attention import (
                GeometricEncoderWrapper,
                HyperbolicAttentionBias,
            )
            self.geo_bias = HyperbolicAttentionBias(
                num_tokens=config.vocab_size,
                embed_dim=hyp_dim,
                c=c,
            )
            self.geo_wrapper = GeometricEncoderWrapper(
                self.encoder,
                geo_bias=self.geo_bias,
            )
            self.geo_wrapper.install_hooks()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through both heads.

        Returns
        -------
        dict with keys:
            logits : (batch, num_classes) from linear head (primary)
            poincare_logits : (batch, num_classes) from Poincaré branch
            distances : (batch, num_classes) geodesic distances
            severity : (batch,) severity scores in [0, 1]
            embedding : (batch, hyp_dim) Poincaré ball embedding
        """
        if self.geo_wrapper is not None:
            self.geo_wrapper.set_input_ids(input_ids)

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = encoder_output.last_hidden_state[:, 0, :]

        # Primary: linear head (with dropout)
        logits = self.linear_head(self.dropout(cls_output))

        # Auxiliary: Poincaré branch (no dropout — has its own projection)
        poincare_out = self.poincare_head(cls_output)

        return {
            "logits": logits,
            "poincare_logits": poincare_out["logits"],
            "distances": poincare_out["distances"],
            "severity": poincare_out["severity"],
            "embedding": poincare_out["embedding"],
        }
