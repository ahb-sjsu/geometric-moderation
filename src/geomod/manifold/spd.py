# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Symmetric Positive Definite (SPD) manifold operations.

SPD matrices naturally represent covariance structures. In content
moderation, they capture how model score distributions vary across
demographic groups — the foundation of geometric fairness auditing.

The SPD manifold S+(d) with the affine-invariant metric is a
Riemannian manifold of non-positive curvature, making geodesic
computations well-posed and unique.
"""

from __future__ import annotations

import torch


class SPDManifold:
    """Operations on the manifold of symmetric positive definite matrices.

    Supports two metrics:
    - Log-Euclidean: faster, approximate
    - Affine-invariant: exact geodesic distance (AIRM)
    """

    @staticmethod
    def log_euclidean_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Log-Euclidean distance: ‖log(A) - log(B)‖_F.

        Fast approximation to the geodesic distance.
        """
        log_A = SPDManifold._matrix_log(A)
        log_B = SPDManifold._matrix_log(B)
        diff = log_A - log_B
        return torch.sqrt((diff * diff).sum(dim=(-2, -1)))

    @staticmethod
    def affine_invariant_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Affine-invariant Riemannian metric (AIRM) distance.

        d(A, B) = ‖log(A^{-1/2} B A^{-1/2})‖_F

        This is the geodesic distance on the SPD manifold.
        """
        A_inv_sqrt = SPDManifold._matrix_power(A, -0.5)
        inner = A_inv_sqrt @ B @ A_inv_sqrt
        log_inner = SPDManifold._matrix_log(inner)
        return torch.sqrt((log_inner * log_inner).sum(dim=(-2, -1)))

    @staticmethod
    def frechet_mean(matrices: list[torch.Tensor], weights: list[float] | None = None,
                     max_iter: int = 50, tol: float = 1e-8) -> torch.Tensor:
        """Fréchet mean on the SPD manifold (iterative algorithm).

        The Fréchet mean minimizes the sum of squared geodesic distances
        to all input matrices.
        """
        n = len(matrices)
        if weights is None:
            weights = [1.0 / n] * n

        # Initialize with arithmetic mean (reasonable starting point)
        mean = sum(w * M for w, M in zip(weights, matrices))

        for _ in range(max_iter):
            # Riemannian gradient step
            mean_inv_sqrt = SPDManifold._matrix_power(mean, -0.5)
            mean_sqrt = SPDManifold._matrix_power(mean, 0.5)

            tangent = torch.zeros_like(mean)
            for w, M in zip(weights, matrices):
                inner = mean_inv_sqrt @ M @ mean_inv_sqrt
                tangent += w * SPDManifold._matrix_log(inner)

            # Check convergence
            if tangent.norm() < tol:
                break

            # Exponential map: new_mean = mean^{1/2} exp(tangent) mean^{1/2}
            exp_tangent = SPDManifold._matrix_exp(tangent)
            mean = mean_sqrt @ exp_tangent @ mean_sqrt

        return mean

    @staticmethod
    def cholesky_param(L: torch.Tensor) -> torch.Tensor:
        """Construct SPD matrix from lower-triangular Cholesky factor.

        S = L L^T guarantees positive definiteness. Useful for
        unconstrained optimization of SPD matrices.
        """
        return L @ L.transpose(-2, -1)

    @staticmethod
    def project(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Project a symmetric matrix onto the SPD cone.

        Clamps eigenvalues to be >= eps.
        """
        eigvals, eigvecs = torch.linalg.eigh(A)
        eigvals = eigvals.clamp(min=eps)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def _matrix_log(A: torch.Tensor) -> torch.Tensor:
        """Matrix logarithm via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(A)
        log_eigvals = torch.log(eigvals.clamp(min=1e-15))
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def _matrix_exp(A: torch.Tensor) -> torch.Tensor:
        """Matrix exponential via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(A)
        exp_eigvals = torch.exp(eigvals)
        return eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)

    @staticmethod
    def _matrix_power(A: torch.Tensor, p: float) -> torch.Tensor:
        """Matrix power A^p via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(A)
        pow_eigvals = eigvals.clamp(min=1e-15).pow(p)
        return eigvecs @ torch.diag_embed(pow_eigvals) @ eigvecs.transpose(-2, -1)
