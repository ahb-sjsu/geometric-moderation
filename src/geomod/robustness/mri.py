# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Manifold Robustness Index (MRI) for content moderation.

Standard adversarial robustness (L_p norms) doesn't capture semantic
robustness. A character substitution that changes "kill" to "k1ll"
is small in L_p but should not change the moderation decision.
Conversely, changing "I'll kill this game" to "I'll kill you" is
small in edit distance but should flip the decision.

MRI operates on the policy manifold:
1. Perturb the input (typos, paraphrase, character subs)
2. Map all perturbations to the Poincaré ball
3. MRI = (spread of perturbation cloud) / (distance to decision boundary)

High MRI → the perturbation cloud stays well within the correct
decision region. Low MRI → perturbations easily cross the boundary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_mri(
    embeddings: torch.Tensor,
    boundary_distances: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Manifold Robustness Index.

    Parameters
    ----------
    embeddings : (n_perturbations, dim)
        Poincaré ball embeddings of the original + perturbed inputs.
    boundary_distances : (n_perturbations,)
        Geodesic distance from each embedding to the nearest decision boundary.

    Returns
    -------
    Scalar MRI value. Higher = more robust.
    """
    # Spread = variance of perturbation cloud
    center = embeddings.mean(dim=0)
    diffs = embeddings - center
    spread = (diffs * diffs).sum(dim=-1).mean()  # mean squared dist from center

    # Minimum boundary distance across perturbations
    min_boundary = boundary_distances.min()

    # MRI = boundary distance / perturbation spread
    return min_boundary / (spread.sqrt() + eps)


def perturbation_cloud(
    model: nn.Module,
    tokenizer,
    original_text: str,
    perturbations: list[str],
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Poincaré ball embeddings for a text and its perturbations.

    Parameters
    ----------
    model : GeometricModerationModel
        Model with hyperbolic classification head.
    tokenizer : HuggingFace tokenizer
    original_text : str
        The original content.
    perturbations : list[str]
        Perturbed versions of the content.

    Returns
    -------
    embeddings : (1 + n_perturbations, hyp_dim)
    logits : (1 + n_perturbations, num_policies)
    """
    all_texts = [original_text] + perturbations

    inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs["embedding"], outputs["logits"]


def text_perturbations(text: str, n: int = 20) -> list[str]:
    """Generate adversarial perturbations of a text.

    Perturbation types:
    1. Character substitution (l→1, o→0, a→@)
    2. Character deletion
    3. Character duplication
    4. Word-level synonym swap (basic)
    """
    import random

    perturbations = []
    chars = list(text)

    # Character substitution map
    leet = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "l": "1", "t": "7"}

    for _ in range(n):
        p_type = random.choice(["sub", "delete", "dup", "swap"])
        p_chars = list(text)

        if p_type == "sub" and len(p_chars) > 0:
            # Random leet-speak substitution
            idx = random.randint(0, len(p_chars) - 1)
            ch = p_chars[idx].lower()
            if ch in leet:
                p_chars[idx] = leet[ch]
            else:
                # Random character nearby on keyboard
                p_chars[idx] = chr(ord(p_chars[idx]) ^ random.choice([1, 2, 32]))

        elif p_type == "delete" and len(p_chars) > 3:
            idx = random.randint(0, len(p_chars) - 1)
            p_chars.pop(idx)

        elif p_type == "dup" and len(p_chars) > 0:
            idx = random.randint(0, len(p_chars) - 1)
            p_chars.insert(idx, p_chars[idx])

        elif p_type == "swap" and len(p_chars) > 1:
            idx = random.randint(0, len(p_chars) - 2)
            p_chars[idx], p_chars[idx + 1] = p_chars[idx + 1], p_chars[idx]

        perturbations.append("".join(p_chars))

    return perturbations
