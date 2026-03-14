# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Hyperbolic content classifier.

Instead of the standard linear → softmax classification head, we:
1. Project the encoder's [CLS] embedding into the Poincaré ball
2. Classify by geodesic distance to policy taxonomy nodes
3. Derive severity from distance to origin

This respects the hierarchical label structure and gives calibrated
severity scores for free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from geomod.manifold.poincare import PoincareBall
from geomod.policy.taxonomy import PolicyNode, PolicyTaxonomyEmbedding


class HyperbolicClassifier(nn.Module):
    """Project encoder output into Poincaré ball, classify by policy distance.

    Parameters
    ----------
    encoder_dim : int
        Dimension of the encoder's [CLS] output (e.g., 768 for BERT-base).
    taxonomy : PolicyNode
        Policy taxonomy tree.
    hyp_dim : int
        Dimension of the Poincaré ball.
    c : float
        Curvature parameter.
    temperature : float
        Softmax temperature for distance-based logits.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        taxonomy: PolicyNode | None = None,
        hyp_dim: int = 32,
        c: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.temperature = temperature

        # Projection: encoder space → tangent space at origin → Poincaré ball
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hyp_dim * 2),
            nn.GELU(),
            nn.Linear(hyp_dim * 2, hyp_dim),
        )
        # Init last layer small so content embeddings start near origin
        # (avoids huge Poincaré distances at init → logit explosion)
        nn.init.normal_(self.projection[-1].weight, std=0.01)
        nn.init.zeros_(self.projection[-1].bias)

        # Policy taxonomy embedding
        if taxonomy is None:
            from geomod.policy.taxonomy import default_taxonomy
            taxonomy = default_taxonomy()
        self.taxonomy_emb = PolicyTaxonomyEmbedding(taxonomy, hyp_dim, c)
        self.taxonomy = taxonomy

    def forward(
        self, encoder_output: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Classify content in hyperbolic space.

        Parameters
        ----------
        encoder_output : (batch, encoder_dim)
            [CLS] token output from the transformer encoder.

        Returns
        -------
        dict with keys:
            - logits: (batch, num_nodes) classification logits
            - distances: (batch, num_nodes) geodesic distances to policy nodes
            - severity: (batch,) severity scores in [0, 1]
            - embedding: (batch, hyp_dim) Poincaré ball embedding
        """
        # Project to tangent space, then exp-map to ball
        tangent = self.projection(encoder_output)
        embedding = self.ball.exp_map_0(tangent)
        embedding = self.ball.project(embedding)

        # Classify by distance to policy nodes
        distances, logits = self.taxonomy_emb.classify(embedding)
        logits = logits / self.temperature

        # Severity = distance from origin
        severity = self.taxonomy_emb.severity(embedding)

        return {
            "logits": logits,
            "distances": distances,
            "severity": severity,
            "embedding": embedding,
        }

    def nearest_policy(self, encoder_output: torch.Tensor, k: int = 3) -> list[list[str]]:
        """Return top-k nearest policy nodes for each item in the batch."""
        result = self.forward(encoder_output)
        distances = result["distances"]  # (batch, num_nodes)

        top_k = distances.topk(k, dim=-1, largest=False)  # smallest distances
        batch_results = []
        for i in range(distances.shape[0]):
            names = [self.taxonomy_emb.node_names[idx] for idx in top_k.indices[i]]
            batch_results.append(names)
        return batch_results


class GeometricModerationModel(nn.Module):
    """Full geometric content moderation model.

    Combines a pretrained transformer encoder with:
    1. Hyperbolic attention bias (optional)
    2. Hyperbolic classification head
    3. Severity scoring

    Parameters
    ----------
    encoder_name : str
        HuggingFace model name (e.g., "microsoft/deberta-v3-base").
    taxonomy : PolicyNode
        Policy taxonomy tree.
    hyp_dim : int
        Hyperbolic embedding dimension.
    c : float
        Curvature.
    use_geometric_attention : bool
        Whether to install hyperbolic attention bias hooks.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        taxonomy: PolicyNode | None = None,
        hyp_dim: int = 32,
        c: float = 1.0,
        use_geometric_attention: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = config.hidden_size

        self.classifier = HyperbolicClassifier(
            encoder_dim=encoder_dim,
            taxonomy=taxonomy,
            hyp_dim=hyp_dim,
            c=c,
            temperature=temperature,
        )

        self.geo_wrapper = None
        self.geo_bias = None
        if use_geometric_attention:
            from geomod.models.attention import GeometricEncoderWrapper, HyperbolicAttentionBias
            # Register geo_bias as a submodule so .to(device) moves it
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
        """Forward pass: encode → project to Poincaré ball → classify."""
        if self.geo_wrapper is not None:
            self.geo_wrapper.set_input_ids(input_ids)

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token output
        cls_output = encoder_output.last_hidden_state[:, 0, :]

        return self.classifier(cls_output)
