"""Genre-adaptive NLI summarization validator package.

A novel system that uses MultiNLI's cross-genre entailment capabilities to validate
and score abstractive summaries for factual consistency.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .models.model import GenreAdaptiveNLIValidator
from .data.loader import SummarizationDataLoader
from .evaluation.metrics import SummaryValidationMetrics
from .exceptions import (
    GenreAdaptiveNLIError,
    ConfigurationError,
    DataLoadError,
    ModelError,
    TrainingError,
    EvaluationError,
)

__all__ = [
    "GenreAdaptiveNLIValidator",
    "SummarizationDataLoader",
    "SummaryValidationMetrics",
    "GenreAdaptiveNLIError",
    "ConfigurationError",
    "DataLoadError",
    "ModelError",
    "TrainingError",
    "EvaluationError",
]