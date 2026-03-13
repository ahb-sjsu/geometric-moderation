# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for Poincaré ball operations."""

import torch
import pytest

from geomod.manifold.poincare import PoincareBall, HyperbolicEmbedding


@pytest.fixture
def ball():
    return PoincareBall(c=1.0)


class TestPoincareBall:
    def test_project_inside(self, ball):
        """Points inside the ball are unchanged."""
        x = torch.tensor([0.1, 0.2, 0.3])
        projected = ball.project(x)
        assert torch.allclose(x, projected)

    def test_project_outside(self, ball):
        """Points outside the ball are projected onto it."""
        x = torch.tensor([10.0, 0.0, 0.0])
        projected = ball.project(x)
        assert projected.norm() < ball.radius

    def test_distance_self_zero(self, ball):
        """Distance from a point to itself is zero."""
        x = torch.tensor([0.1, 0.2])
        d = ball.dist(x, x)
        assert d.item() < 1e-6

    def test_distance_symmetric(self, ball):
        """d(x, y) == d(y, x)."""
        x = torch.randn(5) * 0.3
        y = torch.randn(5) * 0.3
        x, y = ball.project(x), ball.project(y)
        assert torch.allclose(ball.dist(x, y), ball.dist(y, x), atol=1e-5)

    def test_distance_triangle_inequality(self, ball):
        """d(x, z) <= d(x, y) + d(y, z)."""
        x = torch.randn(5) * 0.3
        y = torch.randn(5) * 0.3
        z = torch.randn(5) * 0.3
        x, y, z = ball.project(x), ball.project(y), ball.project(z)
        dxz = ball.dist(x, z)
        dxy = ball.dist(x, y)
        dyz = ball.dist(y, z)
        assert dxz <= dxy + dyz + 1e-5

    def test_distance_increases_near_boundary(self, ball):
        """Points near the boundary should have larger distances."""
        center = torch.tensor([0.1, 0.0])
        near = torch.tensor([0.5, 0.0])
        far = torch.tensor([0.95, 0.0])
        d_near = ball.dist(center, near)
        d_far = ball.dist(center, far)
        assert d_far > d_near

    def test_exp_log_roundtrip(self, ball):
        """exp_map_0(log_map_0(y)) ≈ y for points on the ball."""
        y = torch.randn(5) * 0.3
        y = ball.project(y)
        v = ball.log_map_0(y)
        y_recovered = ball.exp_map_0(v)
        assert torch.allclose(y, y_recovered, atol=1e-5)

    def test_mobius_add_origin(self, ball):
        """x ⊕ 0 = x."""
        x = torch.randn(5) * 0.3
        x = ball.project(x)
        zero = torch.zeros_like(x)
        result = ball.mobius_add(x, zero)
        assert torch.allclose(x, result, atol=1e-6)

    def test_geodesic_endpoints(self, ball):
        """Geodesic at t=0 is x, at t=1 is y."""
        x = torch.randn(5) * 0.2
        y = torch.randn(5) * 0.2
        x, y = ball.project(x), ball.project(y)
        assert torch.allclose(ball.geodesic(x, y, 0.0), x, atol=1e-5)
        assert torch.allclose(ball.geodesic(x, y, 1.0), y, atol=1e-4)

    def test_geodesic_midpoint(self, ball):
        """Midpoint is equidistant from both endpoints."""
        x = torch.randn(5) * 0.2
        y = torch.randn(5) * 0.2
        x, y = ball.project(x), ball.project(y)
        mid = ball.geodesic(x, y, 0.5)
        d_xm = ball.dist(x, mid)
        d_ym = ball.dist(y, mid)
        assert torch.allclose(d_xm, d_ym, atol=1e-3)

    def test_pairwise_dist_shape(self, ball):
        """Pairwise distance matrix has correct shape."""
        x = torch.randn(10, 3) * 0.3
        x = ball.project(x)
        D = ball.pairwise_dist(x)
        assert D.shape == (10, 10)
        # Diagonal should be zero
        assert (D.diag() < 1e-5).all()


class TestHyperbolicEmbedding:
    def test_output_on_ball(self):
        """Embeddings should be inside the Poincaré ball."""
        emb = HyperbolicEmbedding(100, 32, c=1.0)
        ids = torch.arange(10)
        out = emb(ids)
        assert out.shape == (10, 32)
        norms = out.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_pairwise_distances(self):
        """Pairwise distances should be non-negative."""
        emb = HyperbolicEmbedding(20, 8, c=1.0)
        D = emb.pairwise_distances()
        assert D.shape == (20, 20)
        assert (D >= -1e-5).all()
