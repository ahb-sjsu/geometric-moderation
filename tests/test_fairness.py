# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for geometric fairness auditing."""

import torch
import pytest

from geomod.fairness.mahalanobis import (
    group_covariance,
    mahalanobis_fairness_gap,
    mahalanobis_distance_to_group,
)


class TestGroupCovariance:
    def test_basic(self):
        """Covariance matrices are computed per group."""
        scores = torch.randn(100, 5)
        groups = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
        covs = group_covariance(scores, groups)
        assert 0 in covs
        assert 1 in covs
        assert covs[0].shape == (5, 5)

    def test_spd(self):
        """Covariance matrices should be symmetric positive definite."""
        scores = torch.randn(100, 3)
        groups = torch.zeros(100, dtype=torch.long)
        covs = group_covariance(scores, groups)
        C = covs[0]
        # Symmetric
        assert torch.allclose(C, C.T, atol=1e-6)
        # Positive eigenvalues
        eigvals = torch.linalg.eigvalsh(C)
        assert (eigvals > 0).all()

    def test_small_group_skipped(self):
        """Groups below min_group_size are skipped."""
        scores = torch.randn(15, 3)
        groups = torch.cat([torch.zeros(12, dtype=torch.long), torch.ones(3, dtype=torch.long)])
        covs = group_covariance(scores, groups, min_group_size=10)
        assert 0 in covs
        assert 1 not in covs


class TestMahalanobisFairnessGap:
    def test_identical_groups(self):
        """Two groups from the same distribution should have small gap."""
        scores = torch.randn(200, 4)
        groups = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        result = mahalanobis_fairness_gap(scores, groups)
        # Gap should be small (not exactly zero due to sampling)
        assert result["max_gap"] < 2.0

    def test_different_groups(self):
        """Groups with different distributions should have larger gap."""
        group_a = torch.randn(100, 4)
        group_b = torch.randn(100, 4) * 3 + 2  # different scale and mean
        scores = torch.cat([group_a, group_b])
        groups = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        result = mahalanobis_fairness_gap(scores, groups)
        assert result["max_gap"] > 1.0

    def test_reference_group(self):
        """Reference group mode compares all groups to one reference."""
        scores = torch.randn(300, 3)
        groups = torch.cat([
            torch.zeros(100, dtype=torch.long),
            torch.ones(100, dtype=torch.long),
            torch.full((100,), 2, dtype=torch.long),
        ])
        result = mahalanobis_fairness_gap(scores, groups, reference_group=0)
        # Should have comparisons (0,1) and (0,2) only
        assert all(k[0] == 0 for k in result["pairwise"])

    def test_single_group(self):
        """Single group produces zero gap."""
        scores = torch.randn(50, 3)
        groups = torch.zeros(50, dtype=torch.long)
        result = mahalanobis_fairness_gap(scores, groups)
        assert result["max_gap"] == 0.0


class TestMahalanobisDistance:
    def test_center_is_zero(self):
        """Distance from group mean to itself is approximately zero."""
        group = torch.randn(100, 3)
        mean = group.mean(dim=0)
        d = mahalanobis_distance_to_group(mean, group)
        assert d.item() < 0.5  # should be small

    def test_outlier_is_large(self):
        """Distance from an outlier should be large."""
        group = torch.randn(100, 3)
        outlier = torch.tensor([10.0, 10.0, 10.0])
        d = mahalanobis_distance_to_group(outlier, group)
        assert d.item() > 5.0
