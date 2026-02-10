"""Training pipeline for genre-adaptive NLI summarization validator."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from datasets import Dataset as HFDataset
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm
import json

from ..models.model import GenreAdaptiveNLIValidator, GenreAdaptiveNLIConfig
from ..evaluation.metrics import SummaryValidationMetrics
from ..utils.config import Config


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for loss, 'max' for accuracy/F1.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric score.

        Returns:
            True if should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score <= self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == "min"
            if score >= self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        return self.early_stop


class GenreAdaptiveDataset(Dataset):
    """PyTorch dataset wrapper for genre-adaptive training."""

    def __init__(
        self,
        hf_dataset: HFDataset,
        genre_to_id: Dict[str, int]
    ):
        """Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset.
            genre_to_id: Mapping from genre names to IDs.
        """
        self.dataset = hf_dataset
        self.genre_to_id = genre_to_id

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Item index.

        Returns:
            Dictionary with tensors for model input.
        """
        item = self.dataset[idx]

        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

        # Add token type IDs if present
        if "token_type_ids" in item:
            batch["token_type_ids"] = torch.tensor(item["token_type_ids"], dtype=torch.long)

        # Add genre ID
        genre = item.get("genre", "unknown")
        genre_id = self.genre_to_id.get(genre, 0)
        batch["genre_ids"] = torch.tensor(genre_id, dtype=torch.long)

        return batch


class GenreAdaptiveTrainer:
    """Trainer for genre-adaptive NLI models with MLflow integration."""

    def __init__(
        self,
        model: GenreAdaptiveNLIValidator,
        tokenizer: AutoTokenizer,
        config: Config,
        genre_to_id: Dict[str, int]
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            tokenizer: Tokenizer instance.
            config: Configuration object.
            genre_to_id: Genre to ID mapping.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.genre_to_id = genre_to_id
        self.logger = logging.getLogger(__name__)

        # Setup device
        device_config = config.get_device_config()
        self.device = torch.device(device_config["device"])
        self.mixed_precision = device_config["mixed_precision"]

        # Move model to device
        self.model.to(self.device)

        # Setup training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

        # Setup MLflow
        self._setup_mlflow()

        # Metrics calculator
        self.metrics = SummaryValidationMetrics()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_path = None

        # Early stopping
        patience = config.get("training.early_stopping_patience", 3)
        metric_mode = "max" if config.get("training.greater_is_better", True) else "min"
        self.early_stopping = EarlyStopping(patience=patience, mode=metric_mode)

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow_config = self.config.get("mlflow", {})

        # Set tracking URI
        tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        experiment_name = mlflow_config.get("experiment_name", "genre-adaptive-nli")
        mlflow.set_experiment(experiment_name)

    def setup_training(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None
    ) -> None:
        """Setup training components.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset (optional).
        """
        # Create data loaders
        train_pytorch_dataset = GenreAdaptiveDataset(train_dataset, self.genre_to_id)
        self.train_loader = DataLoader(
            train_pytorch_dataset,
            batch_size=self.config.get("training.batch_size", 8),
            shuffle=True,
            num_workers=self.config.get("training.dataloader_num_workers", 4),
            pin_memory=True
        )

        if eval_dataset:
            eval_pytorch_dataset = GenreAdaptiveDataset(eval_dataset, self.genre_to_id)
            self.eval_loader = DataLoader(
                eval_pytorch_dataset,
                batch_size=self.config.get("training.batch_size", 8),
                shuffle=False,
                num_workers=self.config.get("training.dataloader_num_workers", 4),
                pin_memory=True
            )
        else:
            self.eval_loader = None

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

    def _setup_optimizer(self) -> None:
        """Setup optimizer with proper parameter groups."""
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.get("training.weight_decay", 0.01),
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get("training.learning_rate", 2e-5),
            eps=1e-8
        )

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        num_epochs = self.config.get("training.num_epochs", 5)
        num_training_steps = len(self.train_loader) * num_epochs
        warmup_ratio = self.config.get("training.warmup_ratio", 0.1)
        num_warmup_steps = int(warmup_ratio * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset (optional).

        Returns:
            Training history and final metrics.
        """
        self.logger.info("Starting training...")

        # Setup training
        self.setup_training(train_dataset, eval_dataset)

        # Start MLflow run
        with mlflow.start_run():
            # Log configuration
            self._log_config()

            # Training loop
            training_history = []
            num_epochs = self.config.get("training.num_epochs", 5)

            for epoch in range(num_epochs):
                self.epoch = epoch
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")

                # Training step
                train_metrics = self._train_epoch()

                # Evaluation step
                eval_metrics = {}
                if self.eval_loader:
                    eval_metrics = self._evaluate_epoch()

                # Combine metrics
                epoch_metrics = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "eval": eval_metrics
                }
                training_history.append(epoch_metrics)

                # Log to MLflow
                self._log_metrics(epoch_metrics)

                # Save checkpoint
                if self._should_save_checkpoint():
                    self._save_checkpoint(epoch_metrics)

                # Early stopping
                if eval_metrics and self._should_early_stop(eval_metrics):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Save final model
            final_model_path = self._save_final_model()

            # Log model to MLflow
            if self.config.get("mlflow.log_model", True):
                mlflow.pytorch.log_model(self.model, "model")

            return {
                "training_history": training_history,
                "best_model_path": self.best_model_path or final_model_path,
                "final_metrics": training_history[-1] if training_history else {}
            }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Training metrics for the epoch.
        """
        self.model.train()

        total_loss = 0.0
        num_steps = 0
        predictions = []
        labels = []
        genres = []

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Reset gradients
            self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item()
            num_steps += 1
            self.global_step += 1

            # Collect predictions for metrics
            logits = outputs["logits"]
            batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            batch_labels = batch["labels"].cpu().tolist()

            predictions.extend(batch_predictions)
            labels.extend(batch_labels)

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            # Logging
            if self.global_step % self.config.get("training.logging_steps", 100) == 0:
                self.logger.info(f"Step {self.global_step}: loss = {loss.item():.4f}")

        # Calculate epoch metrics
        avg_loss = total_loss / num_steps
        basic_metrics = self.metrics.compute_basic_metrics(predictions, labels)

        epoch_metrics = {
            "loss": avg_loss,
            **basic_metrics
        }

        return epoch_metrics

    def _evaluate_epoch(self) -> Dict[str, float]:
        """Evaluate for one epoch.

        Returns:
            Evaluation metrics for the epoch.
        """
        self.model.eval()

        total_loss = 0.0
        num_steps = 0
        predictions = []
        labels = []
        probabilities = []
        genres = []

        progress_bar = tqdm(self.eval_loader, desc="Evaluation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch_genres = [self._id_to_genre(gid.item()) for gid in batch["genre_ids"]]
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs["loss"]

                # Update metrics
                total_loss += loss.item()
                num_steps += 1

                # Collect predictions for metrics
                logits = outputs["logits"]
                batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                batch_labels = batch["labels"].cpu().tolist()
                batch_probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

                predictions.extend(batch_predictions)
                labels.extend(batch_labels)
                probabilities.extend(batch_probabilities)
                genres.extend(batch_genres)

        # Calculate evaluation metrics
        avg_loss = total_loss / num_steps

        # Basic metrics
        basic_metrics = self.metrics.compute_basic_metrics(predictions, labels, probabilities)

        # Genre-specific metrics
        genre_metrics = self.metrics.compute_genre_specific_metrics(
            predictions, labels, genres, probabilities
        )

        # Target metrics
        target_metrics = self.metrics.compute_target_metrics(
            predictions, labels, genres, probabilities
        )

        # Combine all metrics
        eval_metrics = {
            "loss": avg_loss,
            **basic_metrics,
            **target_metrics
        }

        return eval_metrics

    def _id_to_genre(self, genre_id: int) -> str:
        """Convert genre ID to genre name."""
        id_to_genre = {v: k for k, v in self.genre_to_id.items()}
        return id_to_genre.get(genre_id, "unknown")

    def _log_config(self) -> None:
        """Log configuration to MLflow."""
        # Log training parameters
        mlflow.log_params({
            "model_name": self.model.config.encoder_name,
            "num_epochs": self.config.get("training.num_epochs"),
            "batch_size": self.config.get("training.batch_size"),
            "learning_rate": self.config.get("training.learning_rate"),
            "weight_decay": self.config.get("training.weight_decay"),
            "warmup_ratio": self.config.get("training.warmup_ratio"),
            "num_genres": self.model.config.num_genres,
            "genre_embedding_dim": self.model.config.genre_embedding_dim,
            "genre_adaptation_layers": self.model.config.genre_adaptation_layers,
        })

        # Log tags
        tags = self.config.get("mlflow.tags", {})
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def _log_metrics(self, epoch_metrics: Dict[str, Any]) -> None:
        """Log metrics to MLflow.

        Args:
            epoch_metrics: Metrics for the current epoch.
        """
        epoch = epoch_metrics["epoch"]

        # Log training metrics
        for key, value in epoch_metrics["train"].items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"train_{key}", value, step=epoch)

        # Log evaluation metrics
        for key, value in epoch_metrics["eval"].items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{key}", value, step=epoch)

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        save_steps = self.config.get("training.save_steps", 1000)
        return self.global_step % save_steps == 0

    def _save_checkpoint(self, epoch_metrics: Dict[str, Any]) -> None:
        """Save model checkpoint.

        Args:
            epoch_metrics: Current epoch metrics.
        """
        output_dir = Path(self.config.get("training.output_dir", "./checkpoints"))
        checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metrics
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(epoch_metrics, f, indent=2)

        # Check if this is the best model
        eval_metrics = epoch_metrics.get("eval", {})
        if eval_metrics:
            metric_for_best = self.config.get("training.metric_for_best_model", "entailment_auc")
            current_metric = eval_metrics.get(metric_for_best)

            if current_metric is not None:
                is_better = (
                    self.best_metric is None or
                    (self.config.get("training.greater_is_better", True) and current_metric > self.best_metric) or
                    (not self.config.get("training.greater_is_better", True) and current_metric < self.best_metric)
                )

                if is_better:
                    self.best_metric = current_metric
                    self.best_model_path = str(checkpoint_dir)

                    # Create symlink to best model
                    best_model_link = output_dir / "best-model"
                    if best_model_link.exists():
                        best_model_link.unlink()
                    best_model_link.symlink_to(checkpoint_dir.name)

        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _save_final_model(self) -> str:
        """Save final trained model.

        Returns:
            Path to saved model.
        """
        output_dir = Path(self.config.get("training.output_dir", "./checkpoints"))
        final_model_dir = output_dir / "final-model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)

        self.logger.info(f"Final model saved to {final_model_dir}")
        return str(final_model_dir)

    def _should_early_stop(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early.

        Args:
            eval_metrics: Current evaluation metrics.

        Returns:
            True if should stop, False otherwise.
        """
        metric_for_best = self.config.get("training.metric_for_best_model", "entailment_auc")
        current_metric = eval_metrics.get(metric_for_best)

        if current_metric is None:
            return False

        return self.early_stopping(current_metric)

    def evaluate_model(
        self,
        dataset: HFDataset,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.

        Args:
            dataset: Dataset to evaluate on.
            save_report: Whether to save detailed report.

        Returns:
            Comprehensive evaluation results.
        """
        self.logger.info("Starting comprehensive evaluation...")

        # Create data loader
        pytorch_dataset = GenreAdaptiveDataset(dataset, self.genre_to_id)
        data_loader = DataLoader(
            pytorch_dataset,
            batch_size=self.config.get("training.batch_size", 8),
            shuffle=False,
            num_workers=self.config.get("training.dataloader_num_workers", 4)
        )

        self.model.eval()

        predictions = []
        labels = []
        probabilities = []
        genres = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch_genres = [self._id_to_genre(gid.item()) for gid in batch["genre_ids"]]
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch, return_dict=True)
                logits = outputs["logits"]

                batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                batch_labels = batch["labels"].cpu().tolist()
                batch_probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

                predictions.extend(batch_predictions)
                labels.extend(batch_labels)
                probabilities.extend(batch_probabilities)
                genres.extend(batch_genres)

        # Create comprehensive evaluation report
        output_dir = Path(self.config.get("training.output_dir", "./checkpoints"))
        report_path = output_dir / "evaluation_report.json" if save_report else None

        evaluation_report = self.metrics.create_evaluation_report(
            predictions=predictions,
            labels=labels,
            genres=genres,
            probabilities=probabilities,
            save_path=str(report_path) if report_path else None
        )

        self.logger.info("Evaluation completed")
        return evaluation_report