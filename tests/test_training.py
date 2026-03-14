# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Smoke tests for the Phase 2 training pipeline.

All tests use tiny dimensions (64/16) and synthetic data — no model
downloads or datasets needed.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from geomod.policy.taxonomy import PolicyNode, PolicyTaxonomyEmbedding, default_taxonomy
from geomod.data.datasets import (
    LABEL_TO_TAXONOMY_MAP,
    LABEL_PRIORITY,
    map_scores_to_taxonomy_label,
    map_scores_to_multi_hot,
    map_scores_to_severity,
)
from geomod.training.config import AblationConfig, TrainingConfig
from geomod.training.metrics import (
    compute_metrics,
    compute_severity_calibration,
    compute_ablation_comparison,
)


# ── Taxonomy label mapping tests ─────────────────────────────


class TestLabelMapping:
    @pytest.fixture
    def taxonomy(self):
        return default_taxonomy()

    @pytest.fixture
    def emb(self, taxonomy):
        return PolicyTaxonomyEmbedding(taxonomy, embed_dim=16)

    def test_benign_when_all_below_threshold(self, emb):
        scores = {"toxicity": 0.2, "insult": 0.1, "threat": 0.0}
        label = map_scores_to_taxonomy_label(scores, emb, threshold=0.5)
        assert emb.node_names[label] == "benign"

    def test_highest_priority_wins(self, emb):
        # Both threat and insult above threshold — threat is higher priority
        scores = {"threat": 0.8, "insult": 0.7, "toxicity": 0.6}
        label = map_scores_to_taxonomy_label(scores, emb, threshold=0.5)
        assert emb.node_names[label] == "threats"

    def test_single_active_label(self, emb):
        scores = {"identity_attack": 0.9}
        label = map_scores_to_taxonomy_label(scores, emb, threshold=0.5)
        assert emb.node_names[label] == "hate_speech"

    def test_multi_hot_benign(self, emb):
        scores = {"toxicity": 0.1}
        multi = map_scores_to_multi_hot(scores, emb, threshold=0.5)
        benign_idx = emb.name_to_idx["benign"]
        assert multi[benign_idx] == 1.0
        assert multi.sum() == 1.0

    def test_multi_hot_multiple_active(self, emb):
        scores = {"threat": 0.8, "insult": 0.7}
        multi = map_scores_to_multi_hot(scores, emb, threshold=0.5)
        assert multi[emb.name_to_idx["threats"]] == 1.0
        assert multi[emb.name_to_idx["harassment"]] == 1.0
        assert multi.sum() == 2.0

    def test_severity_benign_is_zero(self, taxonomy, emb):
        scores = {"toxicity": 0.1}
        severity = map_scores_to_severity(scores, taxonomy, emb, threshold=0.5)
        assert severity == 0.0

    def test_severity_reflects_taxonomy(self, taxonomy, emb):
        # severe_toxicity maps to graphic_violence (severity_base=0.7)
        scores = {"severe_toxicity": 0.9}
        severity = map_scores_to_severity(scores, taxonomy, emb, threshold=0.5)
        assert severity == pytest.approx(0.7)

    def test_all_labels_map_to_valid_nodes(self, emb):
        for col, node_name in LABEL_TO_TAXONOMY_MAP.items():
            assert node_name in emb.name_to_idx, f"{col} → {node_name} not in taxonomy"

    def test_priority_order_covers_all_mapped_labels(self):
        for col in LABEL_PRIORITY:
            assert col in LABEL_TO_TAXONOMY_MAP


# ── Config tests ─────────────────────────────────────────────


class TestTrainingConfig:
    def test_default_config(self):
        config = TrainingConfig()
        assert config.ablation == AblationConfig.FULL_GEOMETRIC
        assert config.encoder_lr == 2e-5
        assert config.head_lr == 1e-3
        assert config.taxonomy_lr == 1e-4
        assert config.severity_weight == 0.5

    def test_ablation_enum(self):
        assert AblationConfig.FLAT_BASELINE.value == "flat_baseline"
        assert AblationConfig.HYPERBOLIC_HEAD.value == "hyperbolic_head"
        assert AblationConfig.FULL_GEOMETRIC.value == "full_geometric"

    def test_custom_config(self):
        config = TrainingConfig(
            ablation=AblationConfig.FLAT_BASELINE,
            num_epochs=1,
            max_train_samples=100,
        )
        assert config.ablation == AblationConfig.FLAT_BASELINE
        assert config.num_epochs == 1
        assert config.max_train_samples == 100


# ── Forward/backward pass tests ──────────────────────────────


class TestHyperbolicClassifierPass:
    def test_forward_shape(self):
        from geomod.models.classifier import HyperbolicClassifier

        taxonomy = default_taxonomy()
        num_nodes = len(taxonomy.all_nodes())
        clf = HyperbolicClassifier(
            encoder_dim=64, taxonomy=taxonomy, hyp_dim=16, temperature=0.1
        )

        x = torch.randn(4, 64)
        out = clf(x)

        assert out["logits"].shape == (4, num_nodes)
        assert out["distances"].shape == (4, num_nodes)
        assert out["severity"].shape[0] == 4
        assert out["embedding"].shape == (4, 16)

    def test_backward_pass(self):
        from geomod.models.classifier import HyperbolicClassifier

        taxonomy = default_taxonomy()
        clf = HyperbolicClassifier(
            encoder_dim=64, taxonomy=taxonomy, hyp_dim=16
        )

        x = torch.randn(2, 64)
        out = clf(x)
        loss = out["logits"].sum() + out["severity"].sum()
        loss.backward()

        # Check gradients exist on projection weights
        for p in clf.projection.parameters():
            assert p.grad is not None

    def test_combined_loss(self):
        from geomod.models.classifier import HyperbolicClassifier

        taxonomy = default_taxonomy()
        num_nodes = len(taxonomy.all_nodes())
        clf = HyperbolicClassifier(
            encoder_dim=64, taxonomy=taxonomy, hyp_dim=16
        )

        x = torch.randn(4, 64)
        out = clf(x)

        # Classification loss
        labels = torch.randint(0, num_nodes, (4,))
        ce_loss = torch.nn.functional.cross_entropy(out["logits"], labels)

        # Severity loss
        severity_true = torch.rand(4)
        sev_loss = torch.nn.functional.mse_loss(out["severity"].squeeze(), severity_true)

        total = ce_loss + 0.5 * sev_loss
        total.backward()

        # Taxonomy embeddings should get gradients
        assert clf.taxonomy_emb.embeddings.grad is not None


# ── Metrics tests ────────────────────────────────────────────


class TestMetrics:
    def test_perfect_predictions(self):
        preds = np.array([0, 1, 2, 3, 0, 1])
        labels = np.array([0, 1, 2, 3, 0, 1])
        metrics = compute_metrics(preds, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        preds = rng.randint(0, 5, size=100)
        labels = rng.randint(0, 5, size=100)
        metrics = compute_metrics(preds, labels)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_severity_correlation(self):
        true = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        pred = np.array([0.1, 0.25, 0.45, 0.75, 0.95])
        preds = np.zeros(5, dtype=int)
        labels = np.zeros(5, dtype=int)
        metrics = compute_metrics(preds, labels, severity_pred=pred, severity_true=true)
        assert metrics["severity_spearman"] > 0.9

    def test_severity_calibration(self):
        true = np.random.rand(200)
        pred = true + np.random.randn(200) * 0.05
        pred = np.clip(pred, 0, 1)
        cal = compute_severity_calibration(pred, true, n_bins=5)
        assert cal["ece"] < 0.2
        assert len(cal["bin_edges"]) == 6
        assert len(cal["bin_counts"]) == 5

    def test_ablation_comparison(self):
        results = {
            "flat_baseline": {"accuracy": 0.80, "macro_f1": 0.50},
            "hyperbolic_head": {"accuracy": 0.85, "macro_f1": 0.60},
            "full_geometric": {"accuracy": 0.87, "macro_f1": 0.65},
        }
        comparison = compute_ablation_comparison(results)
        assert "delta_accuracy" not in comparison["flat_baseline"]
        assert comparison["hyperbolic_head"]["delta_accuracy"] == pytest.approx(0.05)
        assert comparison["full_geometric"]["delta_macro_f1"] == pytest.approx(0.15)


# ── Collation tests ──────────────────────────────────────────


class TestCollation:
    def test_collate_shapes(self):
        """Test that collation produces correct tensor shapes."""
        from geomod.data.tokenization import ModerationTokenizer, collate_fn

        taxonomy = default_taxonomy()
        emb = PolicyTaxonomyEmbedding(taxonomy, embed_dim=16)
        num_nodes = emb.num_nodes

        tokenizer = ModerationTokenizer(
            model_name="microsoft/deberta-v3-base",
            max_length=32,
        )

        # Synthetic batch
        batch = [
            {
                "text": "This is a test comment.",
                "taxonomy_label": 0,
                "taxonomy_labels_multi": torch.zeros(num_nodes),
                "severity_target": 0.0,
            },
            {
                "text": "Another comment here.",
                "taxonomy_label": 1,
                "taxonomy_labels_multi": torch.zeros(num_nodes),
                "severity_target": 0.5,
            },
        ]

        result = collate_fn(batch, tokenizer)

        assert result["input_ids"].shape[0] == 2
        assert result["attention_mask"].shape[0] == 2
        assert result["taxonomy_label"].shape == (2,)
        assert result["taxonomy_labels_multi"].shape == (2, num_nodes)
        assert result["severity_target"].shape == (2,)
        assert result["input_ids"].dtype == torch.long
        assert result["severity_target"].dtype == torch.float
