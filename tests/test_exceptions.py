"""Tests for custom exception classes."""

import pytest
from genre_adaptive_nli_summarization_validator.exceptions import (
    GenreAdaptiveNLIError,
    ConfigurationError,
    DataLoadError,
    ModelError,
    TrainingError,
    EvaluationError,
)


class TestGenreAdaptiveNLIError:
    """Tests for the base exception class."""

    def test_basic_exception(self):
        """Test basic exception creation and message."""
        error = GenreAdaptiveNLIError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details is None
        assert error.original_exception is None

    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"key": "value", "number": 42}
        error = GenreAdaptiveNLIError("Test error", details=details)
        assert "Test error" in str(error)
        assert "Details: {'key': 'value', 'number': 42}" in str(error)
        assert error.details == details

    def test_exception_with_original_exception(self):
        """Test exception wrapping another exception."""
        original = ValueError("Original error")
        error = GenreAdaptiveNLIError("Wrapper error", original_exception=original)
        assert "Wrapper error" in str(error)
        assert "Caused by: Original error" in str(error)
        assert error.original_exception == original

    def test_exception_with_all_parameters(self):
        """Test exception with all parameters set."""
        original = RuntimeError("Runtime issue")
        details = {"component": "loader", "step": 3}
        error = GenreAdaptiveNLIError(
            "Complex error",
            details=details,
            original_exception=original
        )

        error_str = str(error)
        assert "Complex error" in error_str
        assert "Details:" in error_str
        assert "Caused by: Runtime issue" in error_str


class TestConfigurationError:
    """Tests for configuration-related errors."""

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from base class."""
        error = ConfigurationError("Config error")
        assert isinstance(error, GenreAdaptiveNLIError)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_with_config_path(self):
        """Test configuration error with config path details."""
        error = ConfigurationError(
            "Invalid configuration",
            details={"config_path": "/path/to/config.yaml"}
        )
        assert "Invalid configuration" in str(error)
        assert "config_path" in str(error)


class TestDataLoadError:
    """Tests for data loading errors."""

    def test_data_load_error_inheritance(self):
        """Test that DataLoadError inherits from base class."""
        error = DataLoadError("Data loading failed")
        assert isinstance(error, GenreAdaptiveNLIError)
        assert isinstance(error, DataLoadError)

    def test_data_load_error_with_dataset_info(self):
        """Test data load error with dataset information."""
        error = DataLoadError(
            "Failed to load dataset",
            details={"dataset": "cnn_dailymail", "split": "train"}
        )
        assert "Failed to load dataset" in str(error)
        assert "cnn_dailymail" in str(error)


class TestModelError:
    """Tests for model-related errors."""

    def test_model_error_inheritance(self):
        """Test that ModelError inherits from base class."""
        error = ModelError("Model initialization failed")
        assert isinstance(error, GenreAdaptiveNLIError)
        assert isinstance(error, ModelError)

    def test_model_error_with_model_info(self):
        """Test model error with model configuration details."""
        error = ModelError(
            "Forward pass failed",
            details={"model_name": "deberta-v3-base", "batch_size": 16}
        )
        assert "Forward pass failed" in str(error)
        assert "deberta-v3-base" in str(error)


class TestTrainingError:
    """Tests for training pipeline errors."""

    def test_training_error_inheritance(self):
        """Test that TrainingError inherits from base class."""
        error = TrainingError("Training failed")
        assert isinstance(error, GenreAdaptiveNLIError)
        assert isinstance(error, TrainingError)

    def test_training_error_with_epoch_info(self):
        """Test training error with epoch information."""
        error = TrainingError(
            "Training epoch failed",
            details={"epoch": 5, "learning_rate": 2e-5}
        )
        assert "Training epoch failed" in str(error)
        assert "epoch" in str(error)


class TestEvaluationError:
    """Tests for evaluation and metrics errors."""

    def test_evaluation_error_inheritance(self):
        """Test that EvaluationError inherits from base class."""
        error = EvaluationError("Metrics computation failed")
        assert isinstance(error, GenreAdaptiveNLIError)
        assert isinstance(error, EvaluationError)

    def test_evaluation_error_with_metric_info(self):
        """Test evaluation error with metric details."""
        error = EvaluationError(
            "AUC calculation failed",
            details={"metric": "auc", "genre": "news", "samples": 1000}
        )
        assert "AUC calculation failed" in str(error)
        assert "metric" in str(error)


class TestExceptionChaining:
    """Tests for exception chaining scenarios."""

    def test_nested_exception_chaining(self):
        """Test chaining multiple levels of exceptions."""
        # Create a nested chain of exceptions
        original = FileNotFoundError("Config file missing")
        config_error = ConfigurationError(
            "Failed to load configuration",
            details={"config_path": "/missing/config.yaml"},
            original_exception=original
        )

        # Wrap in a higher-level error
        model_error = ModelError(
            "Model initialization failed due to config error",
            details={"model_type": "genre-adaptive"},
            original_exception=config_error
        )

        error_str = str(model_error)
        assert "Model initialization failed" in error_str
        assert "Failed to load configuration" in error_str

    def test_exception_with_none_details(self):
        """Test that None details are handled gracefully."""
        error = ConfigurationError("Error message", details=None)
        assert "Error message" in str(error)
        assert "Details: None" not in str(error)

    def test_exception_with_empty_details(self):
        """Test that empty details are handled gracefully."""
        error = DataLoadError("Error message", details={})
        assert "Error message" in str(error)
        assert "Details: {}" in str(error)