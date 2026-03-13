# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for policy taxonomy embedding."""

import torch
import pytest

from geomod.policy.taxonomy import PolicyNode, default_taxonomy, PolicyTaxonomyEmbedding


class TestPolicyNode:
    def test_default_taxonomy_structure(self):
        """Default taxonomy has expected top-level categories."""
        root = default_taxonomy()
        top_names = [c.name for c in root.children]
        assert "violence" in top_names
        assert "hate_speech" in top_names
        assert "sexual_content" in top_names
        assert "benign" in top_names

    def test_all_nodes(self):
        """all_nodes returns correct count."""
        root = default_taxonomy()
        nodes = root.all_nodes()
        assert len(nodes) > 20  # should have plenty of nodes
        assert nodes[0].name == "root"

    def test_find_existing(self):
        """Can find a node by name."""
        root = default_taxonomy()
        node = root.find("slurs")
        assert node is not None
        assert node.name == "slurs"
        assert node.severity_base == 0.9

    def test_find_missing(self):
        """Returns None for non-existent node."""
        root = default_taxonomy()
        assert root.find("nonexistent_category") is None

    def test_depth(self):
        """Taxonomy has reasonable depth."""
        root = default_taxonomy()
        assert root.depth() >= 3  # root → area → category → subcategory

    def test_all_leaves(self):
        """Leaves are nodes with no children."""
        root = default_taxonomy()
        leaves = root.all_leaves()
        for leaf in leaves:
            assert len(leaf.children) == 0


class TestPolicyTaxonomyEmbedding:
    @pytest.fixture
    def emb(self):
        return PolicyTaxonomyEmbedding(default_taxonomy(), embed_dim=16, c=1.0)

    def test_embedding_shape(self, emb):
        """Embeddings have correct shape."""
        all_emb = emb.forward()
        assert all_emb.shape == (emb.num_nodes, 16)

    def test_embeddings_on_ball(self, emb):
        """All embeddings are inside the Poincaré ball."""
        all_emb = emb.forward()
        norms = all_emb.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_root_near_origin(self, emb):
        """Root node should be near the origin."""
        all_emb = emb.forward()
        root_idx = emb.node_index("root")
        root_norm = all_emb[root_idx].norm()
        assert root_norm < 0.1

    def test_classify_shape(self, emb):
        """Classification returns correct shapes."""
        content = torch.randn(4, 16) * 0.1
        content = emb.ball.project(content)
        dists, logits = emb.classify(content)
        assert dists.shape == (4, emb.num_nodes)
        assert logits.shape == (4, emb.num_nodes)

    def test_severity_range(self, emb):
        """Severity should be in [0, 1]."""
        content = torch.randn(10, 16) * 0.3
        content = emb.ball.project(content)
        sev = emb.severity(content)
        assert (sev >= 0).all()
        assert (sev <= 1).all()

    def test_origin_low_severity(self, emb):
        """Content near origin should have low severity."""
        origin = torch.zeros(1, 16)
        sev = emb.severity(origin)
        assert sev.item() < 0.1

    def test_hierarchy_distance(self, emb):
        """Sibling categories should be closer than distant categories."""
        all_emb = emb.forward()
        ball = emb.ball

        # Siblings: slurs and stereotyping (both under racial_hate)
        slurs_idx = emb.node_index("slurs")
        stereo_idx = emb.node_index("stereotyping")
        # Distant: slurs and explicit_sexual
        explicit_idx = emb.node_index("explicit_sexual")

        d_siblings = ball.dist(all_emb[slurs_idx], all_emb[stereo_idx])
        d_distant = ball.dist(all_emb[slurs_idx], all_emb[explicit_idx])

        # This tests the initialization — siblings should start closer
        # (may not hold after training without hierarchy-preserving loss)
        assert d_siblings < d_distant
