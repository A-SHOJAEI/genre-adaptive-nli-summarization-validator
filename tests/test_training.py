"""Tests for training pipeline and related components."""

import pytest
import torch
import tempfile
from pathlib import Path
from datasets import Dataset
import json

from genre_adaptive_nli_summarization_validator.training.trainer import (
    GenreAdaptiveTrainer,
    EarlyStopping,
    GenreAdaptiveDataset
)
from genre_adaptive_nli_summarization_validator.evaluation.metrics import SummaryValidationMetrics


class TestEarlyStopping:
    """Test cases for EarlyStopping utility."""

    def test_init_max_mode(self):
        """Test EarlyStopping initialization in max mode."""
        early_stopping = EarlyStopping(patience=3, mode="max")
        assert early_stopping.patience == 3
        assert early_stopping.mode == "max"
        assert early_stopping.best_score is None
        assert not early_stopping.early_stop

    def test_init_min_mode(self):
        """Test EarlyStopping initialization in min mode."""
        early_stopping = EarlyStopping(patience=5, mode="min")
        assert early_stopping.mode == "min"

    def test_early_stopping_max_mode(self):
        """Test early stopping logic in max mode (for accuracy, F1, etc.)."""
        early_stopping = EarlyStopping(patience=2, mode="max")

        # First score - should not stop
        assert not early_stopping(0.8)
        assert early_stopping.best_score == 0.8

        # Better score - should not stop, reset counter
        assert not early_stopping(0.9)
        assert early_stopping.best_score == 0.9
        assert early_stopping.counter == 0

        # Worse score - increment counter
        assert not early_stopping(0.85)
        assert early_stopping.counter == 1

        # Another worse score - should trigger early stopping
        assert early_stopping(0.8)
        assert early_stopping.early_stop

    def test_early_stopping_min_mode(self):
        """Test early stopping logic in min mode (for loss)."""
        early_stopping = EarlyStopping(patience=2, mode="min")

        # First score
        assert not early_stopping(0.5)
        assert early_stopping.best_score == 0.5

        # Better (lower) score
        assert not early_stopping(0.3)
        assert early_stopping.best_score == 0.3

        # Worse (higher) scores
        assert not early_stopping(0.4)  # Counter = 1
        assert early_stopping(0.5)      # Counter = 2, trigger stop

    def test_min_delta(self):
        """Test minimum delta requirement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode="max")

        assert not early_stopping(0.8)   # Initial
        assert not early_stopping(0.805) # Improvement < min_delta, counter = 1
        assert early_stopping(0.806)     # Still < min_delta, counter = 2, stop


class TestGenreAdaptiveDataset:
    """Test cases for GenreAdaptiveDataset wrapper."""

    def test_init(self, sample_tokenized_data, genre_to_id_mapping):
        """Test dataset initialization."""
        dataset = GenreAdaptiveDataset(sample_tokenized_data, genre_to_id_mapping)
        assert dataset.dataset == sample_tokenized_data
        assert dataset.genre_to_id == genre_to_id_mapping

    def test_len(self, sample_tokenized_data, genre_to_id_mapping):
        """Test dataset length."""
        dataset = GenreAdaptiveDataset(sample_tokenized_data, genre_to_id_mapping)
        assert len(dataset) == len(sample_tokenized_data)

    def test_getitem(self, sample_tokenized_data, genre_to_id_mapping):
        """Test dataset item retrieval."""
        dataset = GenreAdaptiveDataset(sample_tokenized_data, genre_to_id_mapping)

        # Get first item
        item = dataset[0]

        # Check required keys
        required_keys = ["input_ids", "attention_mask", "labels", "genre_ids"]
        for key in required_keys:
            assert key in item

        # Check tensor types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert isinstance(item["genre_ids"], torch.Tensor)

        # Check tensor dtypes
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert item["genre_ids"].dtype == torch.long

    def test_genre_mapping(self, sample_tokenized_data, genre_to_id_mapping):
        """Test genre ID mapping."""
        dataset = GenreAdaptiveDataset(sample_tokenized_data, genre_to_id_mapping)

        for i in range(len(dataset)):
            item = dataset[i]
            original_genre = sample_tokenized_data[i]["genre"]
            expected_id = genre_to_id_mapping.get(original_genre, 0)
            assert item["genre_ids"].item() == expected_id


class TestGenreAdaptiveTrainer:
    """Test cases for GenreAdaptiveTrainer class."""

    @pytest.fixture
    def mock_trainer(self, sample_model, sample_tokenizer, sample_config, genre_to_id_mapping, mock_mlflow_run):
        """Create mock trainer for testing."""
        # Update config for faster testing
        sample_config.set("training.num_epochs", 1)
        sample_config.set("training.batch_size", 2)
        sample_config.set("training.logging_steps", 1)

        trainer = GenreAdaptiveTrainer(
            model=sample_model,
            tokenizer=sample_tokenizer,
            config=sample_config,
            genre_to_id=genre_to_id_mapping
        )
        return trainer

    def test_trainer_init(self, mock_trainer):
        """Test trainer initialization."""
        trainer = mock_trainer

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.config is not None
        assert trainer.genre_to_id is not None
        assert trainer.device is not None
        assert isinstance(trainer.metrics, SummaryValidationMetrics)

    def test_setup_training(self, mock_trainer, sample_tokenized_data):
        """Test training setup."""
        trainer = mock_trainer
        train_dataset = sample_tokenized_data

        trainer.setup_training(train_dataset)

        # Check data loader is created
        assert trainer.train_loader is not None
        assert len(trainer.train_loader) > 0

        # Check optimizer and scheduler are created
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_setup_training_with_eval(self, mock_trainer, sample_tokenized_data):
        """Test training setup with evaluation dataset."""
        trainer = mock_trainer
        train_dataset = sample_tokenized_data
        eval_dataset = sample_tokenized_data  # Reuse for testing

        trainer.setup_training(train_dataset, eval_dataset)

        assert trainer.train_loader is not None
        assert trainer.eval_loader is not None

    def test_train_epoch(self, mock_trainer, sample_tokenized_data, device):
        """Test single training epoch."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Setup training
        trainer.setup_training(sample_tokenized_data)

        # Run one epoch
        train_metrics = trainer._train_epoch()

        # Check metrics returned
        assert isinstance(train_metrics, dict)
        assert "loss" in train_metrics
        assert "accuracy" in train_metrics

        # Check metrics are reasonable
        assert train_metrics["loss"] > 0
        assert 0 <= train_metrics["accuracy"] <= 1

    def test_evaluate_epoch(self, mock_trainer, sample_tokenized_data, device):
        """Test single evaluation epoch."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Setup training with eval
        trainer.setup_training(sample_tokenized_data, sample_tokenized_data)

        # Run evaluation
        eval_metrics = trainer._evaluate_epoch()

        # Check metrics returned
        assert isinstance(eval_metrics, dict)
        assert "loss" in eval_metrics
        assert "accuracy" in eval_metrics

    @pytest.mark.slow
    def test_full_training_loop(self, mock_trainer, sample_tokenized_data, temp_dir, device):
        """Test complete training loop."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Update output directory
        trainer.config.set("training.output_dir", str(temp_dir))

        # Run training
        results = trainer.train(
            train_dataset=sample_tokenized_data,
            eval_dataset=sample_tokenized_data
        )

        # Check results structure
        assert isinstance(results, dict)
        assert "training_history" in results
        assert "best_model_path" in results
        assert "final_metrics" in results

        # Check training history
        history = results["training_history"]
        assert len(history) > 0
        assert "epoch" in history[0]
        assert "train" in history[0]
        assert "eval" in history[0]

    def test_checkpoint_saving(self, mock_trainer, sample_tokenized_data, temp_dir, device):
        """Test checkpoint saving functionality."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Set output directory and force checkpoint saving
        trainer.config.set("training.output_dir", str(temp_dir))
        trainer.config.set("training.save_steps", 1)  # Save every step

        # Setup and run minimal training
        trainer.setup_training(sample_tokenized_data)

        # Create mock epoch metrics
        epoch_metrics = {
            "epoch": 0,
            "train": {"loss": 0.5, "accuracy": 0.8},
            "eval": {"loss": 0.6, "accuracy": 0.75, "entailment_auc": 0.8}
        }

        # Simulate global step
        trainer.global_step = 1

        # Save checkpoint
        trainer._save_checkpoint(epoch_metrics)

        # Check checkpoint directory exists
        checkpoint_dir = temp_dir / f"checkpoint-{trainer.global_step}"
        assert checkpoint_dir.exists()

        # Check required files
        assert (checkpoint_dir / "config.json").exists()
        assert (checkpoint_dir / "pytorch_model.bin").exists()
        assert (checkpoint_dir / "metrics.json").exists()

    def test_model_evaluation(self, mock_trainer, sample_tokenized_data, device):
        """Test comprehensive model evaluation."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Setup training (needed for data loader creation)
        trainer.setup_training(sample_tokenized_data)

        # Run evaluation
        eval_results = trainer.evaluate_model(sample_tokenized_data, save_report=False)

        # Check evaluation results
        assert isinstance(eval_results, dict)
        assert "overall_metrics" in eval_results
        assert "genre_metrics" in eval_results

        # Check overall metrics
        overall = eval_results["overall_metrics"]
        assert "accuracy" in overall
        assert "f1_macro" in overall

    def test_early_stopping_integration(self, mock_trainer, sample_tokenized_data):
        """Test early stopping integration."""
        trainer = mock_trainer

        # Set very low patience for testing
        trainer.config.set("training.early_stopping_patience", 1)

        # Test early stopping decision
        eval_metrics = {"entailment_auc": 0.8}
        should_stop = trainer._should_early_stop(eval_metrics)

        # First call should not stop
        assert not should_stop

        # Second call with same metric should stop (no improvement)
        should_stop = trainer._should_early_stop(eval_metrics)
        assert should_stop

    def test_id_to_genre_mapping(self, mock_trainer):
        """Test genre ID to name conversion."""
        trainer = mock_trainer

        # Test known mapping
        genre_name = trainer._id_to_genre(0)
        assert genre_name in trainer.genre_to_id.values()

        # Test unknown ID
        unknown_genre = trainer._id_to_genre(999)
        assert unknown_genre == "unknown"

    def test_config_logging(self, mock_trainer):
        """Test configuration logging to MLflow."""
        trainer = mock_trainer

        # This should not raise an exception
        try:
            trainer._log_config()
        except Exception as e:
            pytest.fail(f"Config logging failed: {e}")

    def test_metrics_logging(self, mock_trainer):
        """Test metrics logging to MLflow."""
        trainer = mock_trainer

        epoch_metrics = {
            "epoch": 0,
            "train": {"loss": 0.5, "accuracy": 0.8},
            "eval": {"loss": 0.6, "accuracy": 0.75}
        }

        # This should not raise an exception
        try:
            trainer._log_metrics(epoch_metrics)
        except Exception as e:
            pytest.fail(f"Metrics logging failed: {e}")

    def test_optimizer_setup(self, mock_trainer, sample_tokenized_data):
        """Test optimizer setup with parameter groups."""
        trainer = mock_trainer
        trainer.setup_training(sample_tokenized_data)

        optimizer = trainer.optimizer

        # Check optimizer type
        assert optimizer.__class__.__name__ == "AdamW"

        # Check parameter groups (no decay for bias and layer norm)
        assert len(optimizer.param_groups) == 2

        # Check learning rate
        expected_lr = trainer.config.get("training.learning_rate", 2e-5)
        for group in optimizer.param_groups:
            assert group["lr"] == expected_lr

    def test_scheduler_setup(self, mock_trainer, sample_tokenized_data):
        """Test learning rate scheduler setup."""
        trainer = mock_trainer
        trainer.setup_training(sample_tokenized_data)

        scheduler = trainer.scheduler

        # Check scheduler exists
        assert scheduler is not None

        # Check initial learning rate
        initial_lr = scheduler.get_last_lr()[0]
        expected_lr = trainer.config.get("training.learning_rate", 2e-5)
        assert abs(initial_lr - expected_lr) < 1e-8

    def test_mixed_precision_support(self, mock_trainer, sample_tokenized_data, device):
        """Test mixed precision training support."""
        trainer = mock_trainer

        # Check mixed precision setup
        if device.type == "cuda":
            assert trainer.mixed_precision
            assert trainer.scaler is not None
        else:
            # CPU doesn't support mixed precision
            assert not trainer.mixed_precision
            assert trainer.scaler is None

    def test_reproducibility(self, mock_trainer, sample_tokenized_data, device):
        """Test training reproducibility with fixed seeds."""
        trainer1 = mock_trainer
        trainer1.model = trainer1.model.to(device)

        # Run training twice with same seed
        torch.manual_seed(42)
        trainer1.setup_training(sample_tokenized_data)
        metrics1 = trainer1._train_epoch()

        # Reset and run again
        torch.manual_seed(42)
        trainer1.setup_training(sample_tokenized_data)
        metrics2 = trainer1._train_epoch()

        # Results should be similar (allowing for small numerical differences)
        assert abs(metrics1["loss"] - metrics2["loss"]) < 0.1

    def test_gradient_accumulation(self, mock_trainer, sample_tokenized_data, device):
        """Test gradient accumulation functionality."""
        trainer = mock_trainer
        trainer.model = trainer.model.to(device)

        # Set gradient accumulation
        trainer.config.set("training.gradient_accumulation_steps", 2)
        trainer.setup_training(sample_tokenized_data)

        # Run training epoch (should handle gradient accumulation internally)
        train_metrics = trainer._train_epoch()

        # Should complete without errors
        assert "loss" in train_metrics