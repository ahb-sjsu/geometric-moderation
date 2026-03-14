# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Poincaré ball model of hyperbolic space.

Content policy taxonomies are hierarchical:
  policy_area → category → subcategory → specific_rule

Hyperbolic space embeds these trees with O(log n) distortion
(vs. O(n) in Euclidean), making it ideal for representing the
structured label space of content moderation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareBall:
    """Poincaré ball model with curvature -1/c.

    The ball is {x ∈ ℝ^d : c‖x‖² < 1} with conformal factor
    λ_x = 2 / (1 - c‖x‖²).

    Parameters
    ----------
    c : float
        Positive curvature parameter. Radius = 1/√c.
    """

    def __init__(self, c: float = 1.0) -> None:
        self.c = c

    @property
    def radius(self) -> float:
        return 1.0 / self.c**0.5

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition x ⊕_c y."""
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * self.c * xy + self.c * y_sq) * x + (1 - self.c * x_sq) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x_sq * y_sq
        return num / denom.clamp(min=1e-15)

    def exp_map_0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from the origin: tangent vector → ball point."""
        sqrt_c = self.c**0.5
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        return (1.0 / sqrt_c) * torch.tanh(sqrt_c * v_norm) * (v / v_norm)

    def log_map_0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map to the origin: ball point → tangent vector."""
        sqrt_c = self.c**0.5
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15, max=self.radius - 1e-5)
        arg = (sqrt_c * y_norm).clamp(max=1.0 - 1e-6)
        return (1.0 / sqrt_c) * torch.atanh(arg) * (y / y_norm)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at point x: move along tangent vector v."""
        neg_x = self.mobius_add(-x, torch.zeros_like(x))  # -x
        # Transport v from x to origin, exp, transport back
        # Simplified: exp_x(v) = x ⊕ exp_0(v / λ_x) where λ_x is conformal factor
        lam = self._lambda(x)
        v_scaled = v / lam
        return self.mobius_add(x, self.exp_map_0(v_scaled))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Poincaré distance d(x, y)."""
        sqrt_c = self.c**0.5
        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=1e-15, max=self.radius - 1e-5)
        arg = (sqrt_c * diff_norm).clamp(max=1.0 - 1e-6)
        return (2.0 / sqrt_c) * torch.atanh(arg)

    def project(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Project onto the open ball (ensure ‖x‖ < 1/√c)."""
        max_norm = self.radius - eps
        x_norm = x.norm(dim=-1, keepdim=True)
        cond = x_norm > max_norm
        projected = x / x_norm * max_norm
        return torch.where(cond, projected, x)

    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Point at fraction t ∈ [0,1] along the geodesic from x to y."""
        # γ(t) = x ⊕ (t ⊗ (-x ⊕ y))
        diff = self.mobius_add(-x, y)
        scaled = self.mobius_scalar_mul(t, diff)
        return self.mobius_add(x, scaled)

    def mobius_scalar_mul(self, r: float, x: torch.Tensor) -> torch.Tensor:
        """Scalar multiplication r ⊗_c x in the Poincaré ball."""
        sqrt_c = self.c**0.5
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15, max=self.radius - 1e-5)
        arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-6)
        return (1.0 / sqrt_c) * torch.tanh(r * torch.atanh(arg)) * (x / x_norm)

    def pairwise_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Pairwise distance matrix for a set of points.

        Parameters
        ----------
        x : (n, d) tensor of points on the ball

        Returns
        -------
        (n, n) distance matrix
        """
        return self.dist(x.unsqueeze(0), x.unsqueeze(1))

    def _lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor λ_x = 2 / (1 - c‖x‖²)."""
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1 - self.c * x_sq).clamp(min=1e-15)


class HyperbolicEmbedding(nn.Module):
    """Learnable embeddings on the Poincaré ball.

    Maps discrete tokens (policy categories, byte IDs, etc.) to points
    in hyperbolic space. Initialization near the origin ensures
    numerical stability during early training.
    """

    def __init__(self, num_items: int, embed_dim: int, c: float = 1.0) -> None:
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.weight = nn.Parameter(torch.randn(num_items, embed_dim) * 0.05)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up and project embeddings onto the ball."""
        embeds = F.embedding(ids, self.weight)
        return self.ball.project(embeds)

    def all_embeddings(self) -> torch.Tensor:
        """Return all embeddings projected onto the ball."""
        return self.ball.project(self.weight)

    def pairwise_distances(self) -> torch.Tensor:
        """Full pairwise distance matrix between all embeddings."""
        emb = self.all_embeddings()
        return self.ball.pairwise_dist(emb)
