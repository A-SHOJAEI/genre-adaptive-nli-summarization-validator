"""Custom exceptions for the genre-adaptive NLI summarization validator.

This module defines project-specific exception classes that provide clear error
messages and enable precise error handling throughout the application.

Exception Classes:
    GenreAdaptiveNLIError: Base exception for all project-specific errors
    ConfigurationError: Errors related to configuration loading or validation
    DataLoadError: Errors during dataset loading or preprocessing
    ModelError: Errors related to model initialization or forward pass
    TrainingError: Errors during training pipeline execution
    EvaluationError: Errors during metrics computation or evaluation

Example:
    >>> from genre_adaptive_nli_summarization_validator.exceptions import ConfigurationError
    >>> raise ConfigurationError("Invalid configuration file", config_path="/path/to/config")
"""

from typing import Any, Optional


class GenreAdaptiveNLIError(Exception):
    """Base exception class for genre-adaptive NLI summarization validator.

    This serves as the base class for all custom exceptions in the project,
    providing common functionality and ensuring consistent error handling.

    Args:
        message: Human-readable error description
        details: Additional error context or debugging information
        original_exception: Original exception that caused this error (if any)

    Attributes:
        message: The error message
        details: Additional error context
        original_exception: Original exception (if wrapped)
    """

    def __init__(
        self,
        message: str,
        details: Optional[Any] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.details = details
        self.original_exception = original_exception

        # Build comprehensive error message
        error_parts = [message]
        if details:
            error_parts.append(f"Details: {details}")
        if original_exception:
            error_parts.append(f"Caused by: {original_exception}")

        super().__init__(" | ".join(error_parts))


class ConfigurationError(GenreAdaptiveNLIError):
    """Exception raised for configuration-related errors.

    Raised when:
    - Configuration file cannot be loaded or parsed
    - Required configuration keys are missing
    - Configuration values are invalid or out of range
    - Environment variable parsing fails

    Example:
        >>> raise ConfigurationError("Missing required config key", details={"key": "model.name"})
    """
    pass


class DataLoadError(GenreAdaptiveNLIError):
    """Exception raised for data loading and preprocessing errors.

    Raised when:
    - Dataset files cannot be loaded or accessed
    - Data format is invalid or corrupted
    - Required data fields are missing
    - Data preprocessing fails
    - Genre mapping is invalid

    Example:
        >>> raise DataLoadError("Failed to load dataset", details={"dataset": "cnn_dailymail"})
    """
    pass


class ModelError(GenreAdaptiveNLIError):
    """Exception raised for model-related errors.

    Raised when:
    - Model initialization fails
    - Forward pass encounters errors
    - Model configuration is invalid
    - Pre-trained weights cannot be loaded
    - Device placement fails

    Example:
        >>> raise ModelError("Model forward pass failed", details={"batch_size": 16})
    """
    pass


class TrainingError(GenreAdaptiveNLIError):
    """Exception raised for training pipeline errors.

    Raised when:
    - Training loop encounters errors
    - Optimizer setup fails
    - Learning rate scheduler fails
    - Checkpoint saving/loading fails
    - MLflow logging fails
    - Early stopping logic fails

    Example:
        >>> raise TrainingError("Training epoch failed", details={"epoch": 3})
    """
    pass


class EvaluationError(GenreAdaptiveNLIError):
    """Exception raised for evaluation and metrics errors.

    Raised when:
    - Metrics computation fails
    - Evaluation data is invalid
    - Calibration assessment fails
    - Report generation fails
    - Visualization creation fails

    Example:
        >>> raise EvaluationError("Metrics computation failed", details={"metric": "auc"})
    """
    pass