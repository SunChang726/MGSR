import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MGSRTrainer:

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Optional[Dict[str, Any]],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.best_val_score = float("-inf")
        self.global_step = 0
        self.training_stats = defaultdict(list)
        self.validation_stats = defaultdict(list)
        self.logger = self._setup_logger()

    def _create_optimizer(self):
        # Section 4.4: Adam, learning rate 0.0002.
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.0),
        )

    def _create_scheduler(self):
        # The paper uses adaptive learning-rate decay.
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config.get("lr_decay_factor", 0.5),
            patience=self.config.get("lr_decay_patience", 2),
            min_lr=self.config.get("min_lr", 1e-6),
        )

    @staticmethod
    def _setup_logger():
        logger = logging.getLogger("MGSR_Trainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        return logger

    def train(self) -> Dict[str, Any]:
        max_epochs = self.config.get("max_epochs", 20)
        early_stopping_patience = self.config.get("patience", max_epochs)
        patience_counter = 0
        for epoch in range(max_epochs):
            train_metrics = self._train_epoch()
            validation_metrics = self._validate_epoch()
            score = validation_metrics["ndcg@10"]
            self.scheduler.step(score)
            self._record_metrics(train_metrics, validation_metrics)

            is_best = score > self.best_val_score
            if is_best:
                self.best_val_score = score
                patience_counter = 0
            else:
                patience_counter += 1
            self._save_checkpoint(epoch, is_best)
            self.logger.info(
                "epoch=%d train_loss=%.4f ndcg@10=%.4f hr@10=%.4f",
                epoch + 1,
                train_metrics["train_loss"],
                score,
                validation_metrics["hr@10"],
            )
            if patience_counter >= early_stopping_patience:
                break
        return {
            "best_val_score": self.best_val_score,
            "training_stats": dict(self.training_stats),
            "validation_stats": dict(self.validation_stats),
        }

    def _train_epoch(self):
        self.model.train()
        losses = []
        components = defaultdict(list)
        for batch in self.train_dataloader:
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad()
            outputs = self.model(**batch, training_step=self.global_step, mode="train")
            loss = outputs["total_loss"]
            if not loss.requires_grad:
                raise RuntimeError("MGSR total_loss must remain differentiable")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.global_step += 1
            losses.append(loss.detach().item())
            components["gradient_norm"].append(float(gradient_norm))
            for name in ("retrieval_loss", "contrastive_loss", "tokenization_loss"):
                components[name].append(float(outputs[name].detach()))
        if not losses:
            raise ValueError("train_dataloader produced no batches")
        return {
            "train_loss": float(np.mean(losses)),
            **{name: float(np.mean(values)) for name, values in components.items()},
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _validate_epoch(self):
        self.model.eval()
        losses = []
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_batch_to_device(batch)
                outputs = self.model(**batch, mode="inference")
                losses.append(float(outputs["total_loss"]))
                predictions.extend(outputs["final_rankings"])
                targets.extend(batch["target_items"].detach().cpu().tolist())
        metrics = self.compute_ranking_metrics(predictions, targets)
        metrics["val_loss"] = float(np.mean(losses)) if losses else 0.0
        return metrics

    @staticmethod
    def compute_ranking_metrics(predictions, targets):
        if not targets:
            return {"ndcg@10": 0.0, "hr@10": 0.0}
        ndcg = []
        hits = []
        for ranking, target in zip(predictions, targets):
            top = ranking[:10]
            hit = target in top
            hits.append(float(hit))
            ndcg.append(1.0 / np.log2(top.index(target) + 2) if hit else 0.0)
        return {"ndcg@10": float(np.mean(ndcg)), "hr@10": float(np.mean(hits))}

    def _move_batch_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_batch_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_batch_to_device(item) for item in value]
        return value

    def _record_metrics(self, train_metrics, validation_metrics):
        for name, value in train_metrics.items():
            self.training_stats[name].append(value)
        for name, value in validation_metrics.items():
            self.validation_stats[name].append(value)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        checkpoint_dir = self.config.get("checkpoint_dir")
        if not checkpoint_dir:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_score": self.best_val_score,
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_score = checkpoint["best_val_score"]
        return checkpoint


class MultiTaskLoss(nn.Module):

    def forward(self, outputs: Dict[str, Any], targets: Optional[Dict[str, Any]] = None):
        return outputs["total_loss"]


class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            scale = (self.current_epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * scale
        self.current_epoch += 1

    def state_dict(self):
        return {"current_epoch": self.current_epoch, "base_lr": self.base_lr}

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.base_lr = state_dict["base_lr"]


MASRTrainer = MGSRTrainer
