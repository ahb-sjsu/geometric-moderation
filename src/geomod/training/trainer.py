# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Training loop for the geometric content moderation ablation study.

Handles all 3 ablation configurations:
  A: FLAT_BASELINE — DeBERTa + linear head
  B: HYPERBOLIC_HEAD — DeBERTa + hyperbolic classifier
  C: FULL_GEOMETRIC — DeBERTa + geometric attention + hyperbolic classifier

Features:
- Differential learning rates (encoder, head, taxonomy embeddings)
- Combined loss: CrossEntropy (class-weighted) + severity MSE
- Cosine schedule with warmup
- Gradient clipping
- Early stopping on validation macro-F1
- fp16 mixed precision
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from geomod.training.config import AblationConfig, TrainingConfig
from geomod.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _build_model(config: TrainingConfig) -> nn.Module:
    """Build the model for the given ablation configuration."""
    if config.ablation == AblationConfig.FLAT_BASELINE:
        from geomod.models.flat_baseline import FlatModerationModel
        return FlatModerationModel(
            encoder_name=config.encoder_name,
        )
    elif config.ablation == AblationConfig.HYPERBOLIC_HEAD:
        from geomod.models.classifier import GeometricModerationModel
        return GeometricModerationModel(
            encoder_name=config.encoder_name,
            hyp_dim=config.hyp_dim,
            c=config.curvature,
            use_geometric_attention=False,
        )
    elif config.ablation == AblationConfig.FULL_GEOMETRIC:
        from geomod.models.classifier import GeometricModerationModel
        return GeometricModerationModel(
            encoder_name=config.encoder_name,
            hyp_dim=config.hyp_dim,
            c=config.curvature,
            use_geometric_attention=True,
        )
    else:
        raise ValueError(f"Unknown ablation config: {config.ablation}")


def _build_param_groups(model: nn.Module, config: TrainingConfig) -> list[dict]:
    """Build parameter groups with differential learning rates."""
    encoder_params = []
    head_params = []
    taxonomy_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "taxonomy_emb" in name or "geo_bias" in name:
            taxonomy_params.append(param)
        elif "encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": config.encoder_lr,
            "weight_decay": config.weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": config.head_lr,
            "weight_decay": config.weight_decay,
        })
    if taxonomy_params:
        groups.append({
            "params": taxonomy_params,
            "lr": config.taxonomy_lr,
            "weight_decay": 0.0,  # no decay on embeddings
        })
    return groups


def _get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class ModerationTrainer:
    """Training loop for geometric content moderation.

    Parameters
    ----------
    config : TrainingConfig
        Full training configuration.
    train_dataset : Dataset
        Training dataset (CivilCommentsDataset).
    eval_dataset : Dataset
        Evaluation dataset.
    class_weights : torch.Tensor | None
        Inverse-frequency class weights for CE loss.
    label_names : list[str] | None
        Names of taxonomy nodes (for per-class metrics).
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset,
        eval_dataset,
        class_weights: torch.Tensor | None = None,
        label_names: list[str] | None = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.label_names = label_names

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model (ensure float32 — some pretrained weights load as fp16)
        self.model = _build_model(config)
        self.model.float()
        self.model.to(self.device)

        # Losses
        if class_weights is not None and config.use_class_weights:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.severity_loss = nn.MSELoss()

        # Optimizer
        param_groups = _build_param_groups(self.model, config)
        self.optimizer = torch.optim.AdamW(param_groups)

        # Tokenizer + collation
        from geomod.data.tokenization import ModerationTokenizer, make_collate_fn
        self.tokenizer = ModerationTokenizer(
            model_name=config.encoder_name,
            max_length=config.max_length,
        )
        self._collate = make_collate_fn(self.tokenizer)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=0,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=0,
        )

        # Scheduler
        num_training_steps = len(self.train_loader) * config.num_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        self.scheduler = _get_scheduler(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        # Mixed precision
        self.use_amp = config.fp16 and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Early stopping
        self.best_f1 = 0.0
        self.patience_counter = 0

    def _compute_loss(
        self, model_output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute combined classification + severity loss."""
        logits = model_output["logits"]
        labels = batch["taxonomy_label"].to(self.device)

        loss = self.ce_loss(logits, labels)

        if "severity" in model_output and self.config.severity_weight > 0:
            severity_pred = model_output["severity"].squeeze(-1)
            severity_true = batch["severity_target"].to(self.device)
            loss = loss + self.config.severity_weight * self.severity_loss(
                severity_pred, severity_true
            )

        return loss

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self._compute_loss(output, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            # Only step scheduler when optimizer actually updates
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= old_scale:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on the eval set. Returns metrics dict."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_severity_pred = []
        all_severity_true = []

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["taxonomy_label"].numpy()

            all_preds.append(preds)
            all_labels.append(labels)

            if "severity" in output:
                all_severity_pred.append(output["severity"].squeeze(-1).cpu().numpy())
                all_severity_true.append(batch["severity_target"].numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        severity_pred = np.concatenate(all_severity_pred) if all_severity_pred else None
        severity_true = np.concatenate(all_severity_true) if all_severity_true else None

        return compute_metrics(
            all_preds, all_labels,
            severity_pred=severity_pred,
            severity_true=severity_true,
            label_names=self.label_names,
        )

    def train(self) -> dict[str, list]:
        """Full training loop with early stopping.

        Returns
        -------
        dict with training history:
            train_loss : list of per-epoch losses
            eval_metrics : list of per-epoch metric dicts
        """
        history = {"train_loss": [], "eval_metrics": []}

        output_dir = Path(self.config.output_dir) / self.config.ablation.value
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training %s | device=%s | epochs=%d | batch_size=%d",
            self.config.ablation.value,
            self.device,
            self.config.num_epochs,
            self.config.batch_size,
        )

        for epoch in range(self.config.num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch()
            eval_metrics = self.evaluate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["eval_metrics"].append(eval_metrics)

            macro_f1 = eval_metrics.get("macro_f1", 0.0)
            logger.info(
                "Epoch %d/%d | loss=%.4f | macro_f1=%.4f | acc=%.4f | %.1fs",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                macro_f1,
                eval_metrics.get("accuracy", 0.0),
                elapsed,
            )

            # Early stopping
            if macro_f1 > self.best_f1:
                self.best_f1 = macro_f1
                self.patience_counter = 0
                # Save best checkpoint
                torch.save(
                    self.model.state_dict(),
                    output_dir / "best_model.pt",
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return history
