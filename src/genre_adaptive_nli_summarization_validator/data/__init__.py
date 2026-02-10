"""Data loading and preprocessing utilities.

This module provides comprehensive data handling capabilities for the genre-adaptive
NLI summarization validator, including dataset loading, text preprocessing, and
format conversion between summarization and NLI tasks.

Key Features:
    - Multi-domain dataset loading (CNN/DM, XSum, Reddit TIFU)
    - Automatic NLI pair generation from summarization data
    - Cross-genre data splitting and validation
    - Robust text preprocessing with genre-aware tokenization

Classes:
    SummarizationDataLoader: Main data loader supporting multiple summarization
        datasets with automatic NLI conversion.
    TextPreprocessor: Text cleaning and preprocessing utilities with
        configurable pipelines.

Example:
    >>> from genre_adaptive_nli_summarization_validator.data import SummarizationDataLoader
    >>> loader = SummarizationDataLoader(config)
    >>> train_data = loader.load_train_data(['cnn_dailymail', 'xsum'])
"""

from .loader import SummarizationDataLoader
from .preprocessing import TextPreprocessor

__all__ = ["SummarizationDataLoader", "TextPreprocessor"]