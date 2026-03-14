# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Training configuration for the 3-config ablation study.

Config A: FLAT_BASELINE — DeBERTa + linear head (no geometry)
Config B: HYPERBOLIC_HEAD — DeBERTa + hyperbolic classification head
Config C: FULL_GEOMETRIC — DeBERTa + geometric attention + hyperbolic head
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AblationConfig(Enum):
    """Four ablation configurations."""
    FLAT_BASELINE = "flat_baseline"
    HYPERBOLIC_HEAD = "hyperbolic_head"
    FULL_GEOMETRIC = "full_geometric"
    HYBRID = "hybrid"


@dataclass
class TrainingConfig:
    """Full training configuration.

    Parameters
    ----------
    ablation : AblationConfig
        Which model configuration to use.
    encoder_name : str
        HuggingFace encoder model name.
    encoder_lr : float
        Learning rate for the pretrained encoder parameters.
    head_lr : float
        Learning rate for the classification head.
    taxonomy_lr : float
        Learning rate for taxonomy embeddings (Poincaré ball).
    batch_size : int
        Training batch size.
    eval_batch_size : int
        Evaluation batch size (can be larger, no gradients).
    num_epochs : int
        Maximum number of training epochs.
    max_length : int
        Maximum token length for inputs.
    hyp_dim : int
        Poincaré ball dimension for hyperbolic models.
    curvature : float
        Poincaré ball curvature.
    temperature : float
        Temperature for distance-based logits.
    severity_weight : float
        Weight of severity MSE loss relative to classification CE loss.
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    weight_decay : float
        AdamW weight decay.
    max_grad_norm : float
        Gradient clipping norm.
    fp16 : bool
        Use mixed-precision training.
    early_stopping_patience : int
        Stop if val F1 doesn't improve for this many epochs.
    max_train_samples : int | None
        Limit training set size (for debugging).
    max_eval_samples : int | None
        Limit eval set size (for debugging).
    seed : int
        Random seed.
    output_dir : str
        Directory for checkpoints and logs.
    use_class_weights : bool
        Use inverse-frequency class weights in CE loss.
    """
    ablation: AblationConfig = AblationConfig.FULL_GEOMETRIC
    encoder_name: str = "microsoft/deberta-v3-base"
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    taxonomy_lr: float = 1e-4
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 5
    max_length: int = 256
    hyp_dim: int = 32
    curvature: float = 1.0
    temperature: float = 1.0
    auxiliary_weight: float = 0.3
    severity_weight: float = 0.5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    early_stopping_patience: int = 2
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42
    output_dir: str = "outputs/moderation"
    use_class_weights: bool = True
