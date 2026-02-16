"""Training pipeline and optimization components.

This module provides specialized training infrastructure for the genre-adaptive
NLI summarization validator, including custom trainers, early stopping mechanisms,
and cross-genre regularization during training.

Key Features:
    - Genre-aware training loops with domain adaptation
    - Cross-genre regularization loss implementation
    - Early stopping with patience-based validation
    - MLflow integration for experiment tracking
    - Temperature scaling for model calibration

Classes:
    GenreAdaptiveTrainer: Main training class implementing genre-adaptive
        training procedures with cross-domain regularization.

Example:
    >>> from genre_adaptive_nli_summarization_validator.training import GenreAdaptiveTrainer
    >>> trainer = GenreAdaptiveTrainer(model, config)
    >>> trainer.train(train_loader, val_loader)
"""

from .trainer import GenreAdaptiveTrainer

__all__ = ["GenreAdaptiveTrainer"]