# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Mahalanobis-based fairness auditing.

Standard fairness metrics (demographic parity, equalized odds) are
scalars that collapse rich distributional information. Two classifiers
can have identical accuracy parity but very different error *patterns*:

- Classifier A confuses "discussion of racism" with "racial slur"
- Classifier B confuses "suggestive" with "explicit"

Both have the same error rate, but A is far more harmful. Geometric
fairness captures this by comparing the full *distribution* of model
outputs across groups, not just aggregate statistics.

The key insight: compute the covariance matrix of model scores on the
policy manifold for each demographic group, then compare these
covariance matrices using SPD manifold distance. Large SPD distance
between groups means the model behaves qualitatively differently,
even if aggregate metrics look identical.
"""

from __future__ import annotations

import torch
import numpy as np

from geomod.manifold.spd import SPDManifold


def group_covariance(
    scores: torch.Tensor,
    group_labels: torch.Tensor,
    min_group_size: int = 10,
) -> dict[int, torch.Tensor]:
    """Compute covariance matrix of scores for each demographic group.

    Parameters
    ----------
    scores : (n, d) tensor
        Model output scores (d policy dimensions).
    group_labels : (n,) integer tensor
        Demographic group assignment for each sample.
    min_group_size : int
        Minimum samples per group to compute covariance.

    Returns
    -------
    Dict mapping group_id → (d, d) covariance matrix (SPD).
    """
    groups = group_labels.unique()
    covs = {}

    for g in groups:
        mask = group_labels == g.item()
        group_scores = scores[mask]

        if group_scores.shape[0] < min_group_size:
            continue

        # Center
        mean = group_scores.mean(dim=0)
        centered = group_scores - mean

        # Covariance with regularization for positive-definiteness
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        # Regularize: add small diagonal to ensure SPD
        cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)

        covs[g.item()] = cov

    return covs


def mahalanobis_fairness_gap(
    scores: torch.Tensor,
    group_labels: torch.Tensor,
    reference_group: int | None = None,
    metric: str = "log_euclidean",
) -> dict[str, float]:
    """Compute Mahalanobis fairness gap between demographic groups.

    For each pair of groups, compute the SPD manifold distance between
    their score covariance matrices. Large distance = the model's
    output distribution differs qualitatively between groups.

    Parameters
    ----------
    scores : (n, d) model output scores
    group_labels : (n,) group assignments
    reference_group : optional group to compare all others against
    metric : "log_euclidean" or "affine_invariant"

    Returns
    -------
    Dict with:
        - max_gap: largest pairwise SPD distance
        - mean_gap: average pairwise SPD distance
        - pairwise: dict of (g1, g2) → distance
        - group_means: dict of group → mean score vector
    """
    covs = group_covariance(scores, group_labels)
    groups = sorted(covs.keys())

    if len(groups) < 2:
        return {"max_gap": 0.0, "mean_gap": 0.0, "pairwise": {}, "group_means": {}}

    dist_fn = (
        SPDManifold.log_euclidean_dist
        if metric == "log_euclidean"
        else SPDManifold.affine_invariant_dist
    )

    pairwise = {}
    if reference_group is not None and reference_group in covs:
        # Compare all groups to reference
        ref_cov = covs[reference_group]
        for g in groups:
            if g == reference_group:
                continue
            d = dist_fn(ref_cov, covs[g]).item()
            pairwise[(reference_group, g)] = d
    else:
        # All pairs
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                d = dist_fn(covs[g1], covs[g2]).item()
                pairwise[(g1, g2)] = d

    distances = list(pairwise.values())

    # Group means
    group_means = {}
    for g in groups:
        mask = group_labels == g
        group_means[g] = scores[mask].mean(dim=0).tolist()

    return {
        "max_gap": max(distances) if distances else 0.0,
        "mean_gap": float(np.mean(distances)) if distances else 0.0,
        "pairwise": pairwise,
        "group_means": group_means,
    }


def mahalanobis_distance_to_group(
    point: torch.Tensor,
    group_scores: torch.Tensor,
) -> torch.Tensor:
    """Mahalanobis distance from a point to a group distribution.

    d_M(x, G) = sqrt((x - μ_G)^T Σ_G^{-1} (x - μ_G))

    This measures how "typical" a score is for a given group.
    Content flagged as harmful for one group but not another has
    different Mahalanobis distances to each group's distribution.
    """
    mean = group_scores.mean(dim=0)
    centered = group_scores - mean
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)

    diff = point - mean
    cov_inv = torch.linalg.inv(cov)
    return torch.sqrt(diff @ cov_inv @ diff)
