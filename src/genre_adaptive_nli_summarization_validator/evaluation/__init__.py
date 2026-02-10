"""Comprehensive evaluation and metrics computation.

This module provides extensive evaluation capabilities for the genre-adaptive
NLI summarization validator, including specialized metrics for cross-genre
performance assessment and hallucination detection.

Key Features:
    - Multi-domain performance evaluation
    - Calibration assessment and reliability metrics
    - Hallucination detection scoring
    - Genre transfer analysis
    - Comprehensive reporting with visualizations

Classes:
    SummaryValidationMetrics: Comprehensive metrics suite for evaluating
        genre-adaptive summarization validation performance.

Example:
    >>> from genre_adaptive_nli_summarization_validator.evaluation import SummaryValidationMetrics
    >>> metrics = SummaryValidationMetrics(config)
    >>> results = metrics.compute_all_metrics(predictions, labels, genres)
"""

from .metrics import SummaryValidationMetrics

__all__ = ["SummaryValidationMetrics"]