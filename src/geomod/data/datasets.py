# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Civil Comments dataset with taxonomy label mapping.

Maps the 7 continuous toxicity scores from Civil Comments into
the policy taxonomy node indices for geometric classification.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

import numpy as np

from geomod.policy.taxonomy import PolicyNode, PolicyTaxonomyEmbedding, default_taxonomy

CIVIL_COMMENTS_LABEL_COLS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# Maps civil_comments columns → taxonomy node names.
# When multiple labels are active, priority ordering picks the most severe.
LABEL_TO_TAXONOMY_MAP = {
    "severe_toxicity": "graphic_violence",
    "threat": "threats",
    "identity_attack": "hate_speech",
    "sexual_explicit": "sexual_content",
    "obscene": "explicit_sexual",
    "insult": "harassment",
    "toxicity": "violence",
}

# Priority order: most specific/severe first
LABEL_PRIORITY = [
    "severe_toxicity",
    "threat",
    "identity_attack",
    "sexual_explicit",
    "obscene",
    "insult",
    "toxicity",
]


def map_scores_to_taxonomy_label(
    scores: dict[str, float],
    taxonomy_emb: PolicyTaxonomyEmbedding,
    threshold: float = 0.5,
) -> int:
    """Map continuous scores to a single taxonomy node index.

    Returns the benign index if no score exceeds the threshold,
    otherwise returns the highest-priority active label's taxonomy node.
    """
    for col in LABEL_PRIORITY:
        if scores.get(col, 0.0) >= threshold:
            node_name = LABEL_TO_TAXONOMY_MAP[col]
            if node_name in taxonomy_emb.name_to_idx:
                return taxonomy_emb.name_to_idx[node_name]
    return taxonomy_emb.name_to_idx["benign"]


def map_scores_to_multi_hot(
    scores: dict[str, float],
    taxonomy_emb: PolicyTaxonomyEmbedding,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Map continuous scores to a multi-hot vector over taxonomy nodes."""
    multi_hot = torch.zeros(taxonomy_emb.num_nodes)
    any_active = False
    for col in LABEL_TO_TAXONOMY_MAP:
        if scores.get(col, 0.0) >= threshold:
            node_name = LABEL_TO_TAXONOMY_MAP[col]
            if node_name in taxonomy_emb.name_to_idx:
                multi_hot[taxonomy_emb.name_to_idx[node_name]] = 1.0
                any_active = True
    if not any_active:
        multi_hot[taxonomy_emb.name_to_idx["benign"]] = 1.0
    return multi_hot


def map_scores_to_severity(
    scores: dict[str, float],
    taxonomy: PolicyNode,
    taxonomy_emb: PolicyTaxonomyEmbedding,
    threshold: float = 0.5,
) -> float:
    """Derive severity target from active labels' taxonomy severity_base."""
    max_severity = 0.0
    for col in LABEL_TO_TAXONOMY_MAP:
        if scores.get(col, 0.0) >= threshold:
            node_name = LABEL_TO_TAXONOMY_MAP[col]
            node = taxonomy.find(node_name)
            if node is not None:
                max_severity = max(max_severity, node.severity_base)
    return max_severity


class CivilCommentsDataset(Dataset):
    """Civil Comments dataset mapped to the policy taxonomy.

    Each item returns:
        text: str
        taxonomy_label: int (single-label index)
        taxonomy_labels_multi: Tensor (multi-hot over taxonomy nodes)
        severity_target: float
    """

    def __init__(
        self,
        split: str = "train",
        taxonomy: PolicyNode | None = None,
        threshold: float = 0.5,
        max_samples: int | None = None,
        cache_dir: str | None = None,
    ):
        from datasets import load_dataset

        self.taxonomy = taxonomy or default_taxonomy()
        self.taxonomy_emb = PolicyTaxonomyEmbedding(self.taxonomy)
        self.threshold = threshold

        ds = load_dataset("civil_comments", split=split, cache_dir=cache_dir)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        text = row["text"]
        scores = {col: row.get(col, 0.0) for col in CIVIL_COMMENTS_LABEL_COLS}

        taxonomy_label = map_scores_to_taxonomy_label(
            scores, self.taxonomy_emb, self.threshold
        )
        taxonomy_labels_multi = map_scores_to_multi_hot(
            scores, self.taxonomy_emb, self.threshold
        )
        severity_target = map_scores_to_severity(
            scores, self.taxonomy, self.taxonomy_emb, self.threshold
        )

        return {
            "text": text,
            "taxonomy_label": taxonomy_label,
            "taxonomy_labels_multi": taxonomy_labels_multi,
            "severity_target": severity_target,
        }


def load_civil_comments(
    split: str = "train",
    taxonomy: PolicyNode | None = None,
    threshold: float = 0.5,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> CivilCommentsDataset:
    """Convenience loader for Civil Comments."""
    return CivilCommentsDataset(
        split=split,
        taxonomy=taxonomy,
        threshold=threshold,
        max_samples=max_samples,
        cache_dir=cache_dir,
    )


def get_label_weights(dataset: CivilCommentsDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced labels.

    Returns a weight tensor of shape (num_nodes,) suitable for
    CrossEntropyLoss(weight=...).
    """
    num_nodes = dataset.taxonomy_emb.num_nodes
    counts = torch.zeros(num_nodes)

    for i in range(len(dataset)):
        label = dataset[i]["taxonomy_label"]
        counts[label] += 1

    # Inverse frequency with smoothing
    total = counts.sum()
    weights = total / (num_nodes * counts.clamp(min=1))
    # Cap extreme weights
    weights = weights.clamp(max=100.0)
    return weights
