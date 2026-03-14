# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Content policy taxonomy → Poincaré ball embedding.

A policy taxonomy is a tree:
    root → policy_area → category → subcategory → specific_rule

We embed this tree into the Poincaré ball such that:
- Root sits at the origin (most general)
- Depth ≈ distance from origin (more specific = farther out)
- Sibling distance ≈ semantic relatedness
- Geodesic between two nodes follows the tree path

This gives the classifier a structured prior over the label space:
"borderline" content sits geometrically between categories, and
severity is naturally encoded as distance from origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from geomod.manifold.poincare import PoincareBall


@dataclass
class PolicyNode:
    """A node in the content policy taxonomy."""
    name: str
    children: list[PolicyNode] = field(default_factory=list)
    description: str = ""
    severity_base: float = 0.0  # base severity [0, 1]

    def all_leaves(self) -> list[PolicyNode]:
        """Return all leaf nodes (specific rules)."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def all_nodes(self) -> list[PolicyNode]:
        """Return all nodes in BFS order."""
        result = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    def depth(self) -> int:
        """Maximum depth of the subtree."""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def find(self, name: str) -> PolicyNode | None:
        """Find a node by name."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find(name)
            if found:
                return found
        return None


def default_taxonomy() -> PolicyNode:
    """Standard content moderation policy taxonomy.

    Based on common industry policies (OpenAI, Meta, Google).
    """
    return PolicyNode("root", children=[
        PolicyNode("violence", severity_base=0.6, children=[
            PolicyNode("threats", severity_base=0.8, children=[
                PolicyNode("direct_threats", severity_base=0.95),
                PolicyNode("indirect_threats", severity_base=0.7),
                PolicyNode("conditional_threats", severity_base=0.6),
            ]),
            PolicyNode("graphic_violence", severity_base=0.7, children=[
                PolicyNode("real_violence", severity_base=0.9),
                PolicyNode("fictional_violence", severity_base=0.4),
                PolicyNode("news_reporting", severity_base=0.2),
            ]),
            PolicyNode("self_harm", severity_base=0.8, children=[
                PolicyNode("self_harm_promotion", severity_base=0.95),
                PolicyNode("self_harm_discussion", severity_base=0.3),
                PolicyNode("self_harm_resources", severity_base=0.1),
            ]),
        ]),
        PolicyNode("hate_speech", severity_base=0.7, children=[
            PolicyNode("racial_hate", severity_base=0.8, children=[
                PolicyNode("slurs", severity_base=0.9),
                PolicyNode("stereotyping", severity_base=0.6),
                PolicyNode("historical_discussion", severity_base=0.1),
            ]),
            PolicyNode("gender_hate", severity_base=0.7, children=[
                PolicyNode("misogyny", severity_base=0.8),
                PolicyNode("misandry", severity_base=0.7),
                PolicyNode("transphobia", severity_base=0.8),
            ]),
            PolicyNode("religious_hate", severity_base=0.7, children=[
                PolicyNode("blasphemy_attacks", severity_base=0.7),
                PolicyNode("theological_debate", severity_base=0.1),
            ]),
            PolicyNode("disability_hate", severity_base=0.7),
        ]),
        PolicyNode("sexual_content", severity_base=0.5, children=[
            PolicyNode("explicit_sexual", severity_base=0.8),
            PolicyNode("suggestive", severity_base=0.4),
            PolicyNode("educational_sexual", severity_base=0.1),
        ]),
        PolicyNode("harassment", severity_base=0.6, children=[
            PolicyNode("targeted_harassment", severity_base=0.8),
            PolicyNode("bullying", severity_base=0.7),
            PolicyNode("doxxing", severity_base=0.9),
        ]),
        PolicyNode("misinformation", severity_base=0.5, children=[
            PolicyNode("health_misinfo", severity_base=0.7),
            PolicyNode("political_misinfo", severity_base=0.6),
            PolicyNode("conspiracy_theories", severity_base=0.4),
        ]),
        PolicyNode("csam", severity_base=1.0, description="Zero tolerance"),
        PolicyNode("benign", severity_base=0.0),
    ])


class PolicyTaxonomyEmbedding(nn.Module):
    """Embed a policy taxonomy tree into the Poincaré ball.

    The embedding is initialized using the tree structure (depth → radius,
    sibling index → angle) and then refined during training.

    The resulting embedding provides:
    1. Distance-based classification (nearest policy node)
    2. Severity scores (geodesic distance from origin)
    3. Uncertainty quantification (distance to decision boundary)
    """

    def __init__(self, taxonomy: PolicyNode, embed_dim: int = 32, c: float = 1.0) -> None:
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.taxonomy = taxonomy
        self.embed_dim = embed_dim

        # Enumerate all nodes
        self.nodes = taxonomy.all_nodes()
        self.node_names = [n.name for n in self.nodes]
        self.name_to_idx = {n.name: i for i, n in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)

        # Initialize embeddings from tree structure
        init = self._tree_init(taxonomy, embed_dim)
        self.embeddings = nn.Parameter(init)

    def _tree_init(self, root: PolicyNode, dim: int) -> torch.Tensor:
        """Initialize Poincaré embeddings from tree geometry.

        Depth → radial distance (deeper = farther from origin)
        Sibling index → angular position (spread around parent)
        """
        max_depth = root.depth()
        init = torch.zeros(self.num_nodes, dim)

        def _assign(node: PolicyNode, depth: int, angle_start: float, angle_span: float):
            idx = self.name_to_idx[node.name]

            if depth == 0:
                # Root at origin
                init[idx] = torch.zeros(dim)
            else:
                # Radius proportional to depth — spread nodes across the ball
                # so distance-based classification has meaningful gradients
                radius = 0.5 * depth / max(max_depth, 1)
                # Angle in 2D subspace (first two dims)
                angle = angle_start + angle_span / 2
                init[idx, 0] = radius * torch.cos(torch.tensor(angle))
                init[idx, 1] = radius * torch.sin(torch.tensor(angle))
                # Small random perturbation in remaining dims
                if dim > 2:
                    init[idx, 2:] = torch.randn(dim - 2) * 0.01

            # Recurse to children
            n_children = len(node.children)
            if n_children > 0:
                child_span = angle_span / n_children
                for i, child in enumerate(node.children):
                    child_start = angle_start + i * child_span
                    _assign(child, depth + 1, child_start, child_span)

        import math
        _assign(root, 0, 0.0, 2 * math.pi)
        return init

    def forward(self) -> torch.Tensor:
        """Return all policy node embeddings projected onto the ball."""
        return self.ball.project(self.embeddings)

    def classify(self, content_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Classify content by nearest policy node (geodesic distance).

        Parameters
        ----------
        content_embedding : (batch, dim) tensor in the Poincaré ball

        Returns
        -------
        distances : (batch, num_nodes) geodesic distances
        logits : (batch, num_nodes) negative distances (for softmax)
        """
        policy_embs = self.forward()  # (num_nodes, dim)
        # Pairwise distances: (batch, num_nodes)
        dists = self.ball.dist(
            content_embedding.unsqueeze(1),  # (batch, 1, dim)
            policy_embs.unsqueeze(0),         # (1, num_nodes, dim)
        )
        return dists, -dists

    def severity(self, content_embedding: torch.Tensor) -> torch.Tensor:
        """Compute severity as geodesic distance from origin.

        Content near the origin = benign. Content far from origin = severe.
        Normalized to [0, 1] via tanh.
        """
        origin = torch.zeros_like(content_embedding[:, :1].expand_as(content_embedding))
        raw_dist = self.ball.dist(content_embedding, origin)
        return torch.tanh(raw_dist)  # normalize to [0, 1]

    def node_index(self, name: str) -> int:
        """Get the index of a policy node by name."""
        return self.name_to_idx[name]
