"""Genre-adaptive NLI model implementations.

This module contains the core neural network architectures for the genre-adaptive
NLI summarization validator, including the main model class and associated components.

The primary model leverages DeBERTa-v3 as a base with custom genre adaptation layers
that enable cross-domain transferability for summarization validation tasks.

Classes:
    GenreAdaptiveNLIValidator: Main model class implementing genre-adaptive
        entailment classification for summary validation.

Example:
    >>> from genre_adaptive_nli_summarization_validator.models import GenreAdaptiveNLIValidator
    >>> model = GenreAdaptiveNLIValidator.from_pretrained('path/to/checkpoint')
    >>> predictions = model(premise, hypothesis, genre_ids)
"""

from .model import GenreAdaptiveNLIValidator

__all__ = ["GenreAdaptiveNLIValidator"]