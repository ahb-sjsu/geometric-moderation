# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Hyperbolic attention bias for transformer encoders.

Injects policy-hierarchy-aware bias into self-attention via forward
hooks. Tokens semantically close in the policy taxonomy attend more
to each other, giving the model a structured prior over content
categories without modifying pretrained weights.

This is architecturally identical to the geometric attention in
deep-past (Akkadian cuneiform MT), but here the bias encodes
content policy hierarchy rather than cuneiform sign hierarchy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomod.manifold.poincare import PoincareBall


class HyperbolicAttentionBias(nn.Module):
    """Compute attention bias from hyperbolic token embeddings.

    Each token position gets a learnable embedding on the Poincaré ball.
    The bias for position (i, j) is α · (-d(e_i, e_j)), so nearby tokens
    get higher attention.

    Parameters
    ----------
    num_tokens : int
        Vocabulary size (e.g., 30522 for BERT).
    embed_dim : int
        Dimension of hyperbolic embeddings.
    c : float
        Poincaré ball curvature.
    """

    def __init__(self, num_tokens: int = 30522, embed_dim: int = 32, c: float = 1.0) -> None:
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.weight = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.05)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute attention bias matrix.

        Parameters
        ----------
        token_ids : (seq_len,) or (batch, seq_len) token IDs

        Returns
        -------
        (seq_len, seq_len) or (batch, seq_len, seq_len) attention bias
        """
        token_ids = token_ids.clamp(0, self.weight.shape[0] - 1)
        embeds = F.embedding(token_ids, self.weight)
        embeds = self.ball.project(embeds)
        # Pairwise distance → negative = similarity
        dists = self.ball.dist(embeds.unsqueeze(-2), embeds.unsqueeze(-3))
        return self.scale * (-dists)


class GeometricEncoderWrapper:
    """Wraps a HuggingFace transformer encoder with hyperbolic attention bias.

    Installs forward hooks on the encoder's self-attention layers to
    add geometric bias to the position bias / attention scores.

    Works with BERT, RoBERTa, DeBERTa, and other HuggingFace encoders.

    Parameters
    ----------
    model : nn.Module
        HuggingFace model (e.g., BertForSequenceClassification).
    vocab_size : int
        Model vocabulary size.
    embed_dim : int
        Hyperbolic embedding dimension.
    c : float
        Poincaré ball curvature.
    num_layers_bias : int
        Number of encoder layers to inject bias into (from the bottom).
    """

    def __init__(
        self,
        model: nn.Module,
        geo_bias: HyperbolicAttentionBias | None = None,
        vocab_size: int = 30522,
        embed_dim: int = 32,
        c: float = 1.0,
        num_layers_bias: int = 4,
    ) -> None:
        self.model = model
        if geo_bias is not None:
            self.geo_bias = geo_bias
        else:
            self.geo_bias = HyperbolicAttentionBias(
                num_tokens=vocab_size,
                embed_dim=embed_dim,
                c=c,
            )
        self.num_layers_bias = num_layers_bias
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._current_input_ids: torch.Tensor | None = None

    def install_hooks(self) -> None:
        """Install forward hooks on encoder self-attention layers."""
        self.remove_hooks()
        encoder_layers = self._find_encoder_layers()
        n = min(self.num_layers_bias, len(encoder_layers))

        for i in range(n):
            attn = self._find_self_attention(encoder_layers[i])
            if attn is not None:
                hook = attn.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)

        print(f"Geometric bias hooks installed on {len(self._hooks)} encoder layers")

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        """Set current input IDs for the hooks to use."""
        self._current_input_ids = input_ids

    def parameters(self) -> list[nn.Parameter]:
        """All trainable parameters: model + geometric bias."""
        return list(self.model.parameters()) + list(self.geo_bias.parameters())

    def _attention_hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        """Forward hook that adds geometric bias to attention output.

        We add the geometric bias as a residual attention path:
        compute geo-attention weights from hyperbolic distances,
        apply them to the VALUE vectors (from the attention output),
        and add as a scaled residual.
        """
        if self._current_input_ids is None:
            return output

        if isinstance(output, tuple) and len(output) >= 1:
            attn_output = output[0]
            # Compute geometric attention bias (negative distances = similarity)
            geo_b = self.geo_bias(self._current_input_ids)  # (batch, seq, seq)
            geo_b = geo_b.to(dtype=attn_output.dtype, device=attn_output.device)
            seq_len = min(geo_b.shape[-1], attn_output.shape[1])
            geo_b = geo_b[:, :seq_len, :seq_len]

            # Scale matters: geo_b values should be comparable to attention logits
            # Use the learned scale parameter inside geo_bias
            geo_weights = torch.softmax(geo_b, dim=-1)  # (batch, seq, seq)
            geo_ctx = torch.matmul(geo_weights, attn_output[:, :seq_len, :])

            # Gated residual: let model learn how much geometric info to use
            modified = attn_output.clone()
            modified[:, :seq_len, :] = attn_output[:, :seq_len, :] + geo_ctx
            return (modified,) + output[1:]

        return output

    def _find_encoder_layers(self) -> nn.ModuleList:
        """Find the encoder layer list in a HuggingFace model."""
        # Try common attribute paths
        for path in [
            "encoder.layer",      # BERT, RoBERTa
            "encoder.block",      # T5
            "transformer.layer",  # DistilBERT
            "deberta.encoder.layer",  # DeBERTa
        ]:
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue

        raise ValueError(
            f"Cannot find encoder layers in {type(self.model).__name__}. "
            "Supported: BERT, RoBERTa, DeBERTa, T5, DistilBERT."
        )

    def _find_self_attention(self, layer: nn.Module) -> nn.Module | None:
        """Find the self-attention module within an encoder layer."""
        for name in ["attention.self", "attention", "SelfAttention", "self_attn"]:
            obj = layer
            try:
                for attr in name.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        return None
