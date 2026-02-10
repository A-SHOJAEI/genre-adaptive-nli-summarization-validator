"""Test configuration and fixtures for genre-adaptive NLI summarization validator."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer
from datasets import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genre_adaptive_nli_summarization_validator.models.model import (
    GenreAdaptiveNLIValidator,
    GenreAdaptiveNLIConfig
)
from genre_adaptive_nli_summarization_validator.data.loader import SummarizationDataLoader
from genre_adaptive_nli_summarization_validator.data.preprocessing import TextPreprocessor
from genre_adaptive_nli_summarization_validator.utils.config import Config


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    config_data = {
        "model": {
            "name": "microsoft/deberta-v3-base",
            "max_length": 128,  # Smaller for testing
            "dropout": 0.1,
            "num_labels": 3,
            "genre_adaptation_layers": 1,  # Smaller for testing
            "genre_embedding_dim": 64,  # Smaller for testing
        },
        "training": {
            "batch_size": 2,  # Small for testing
            "num_epochs": 2,
            "learning_rate": 2e-5,
            "seed": 42,
        },
        "data": {
            "cache_dir": "./test_cache",
            "max_train_samples": 100,
            "max_val_samples": 50,
        }
    }

    # Create temporary config
    import yaml
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_data, temp_config, default_flow_style=False)
    temp_config.close()

    config = Config(temp_config.name)
    yield config

    # Cleanup
    Path(temp_config.name).unlink()


@pytest.fixture
def sample_tokenizer():
    """Create sample tokenizer for testing."""
    # Use a lightweight tokenizer for testing
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


@pytest.fixture
def sample_text_preprocessor():
    """Create sample text preprocessor for testing."""
    return TextPreprocessor("microsoft/deberta-v3-base")


@pytest.fixture
def genre_to_id_mapping():
    """Create sample genre to ID mapping."""
    return {
        "fiction": 0,
        "news": 1,
        "academic": 2,
        "government": 3,
        "unknown": 4
    }


@pytest.fixture
def sample_model_config(genre_to_id_mapping):
    """Create sample model configuration."""
    return GenreAdaptiveNLIConfig(
        base_model_name="microsoft/deberta-v3-base",
        num_labels=3,
        num_genres=len(genre_to_id_mapping),
        genre_embedding_dim=64,  # Smaller for testing
        genre_adaptation_layers=1,  # Smaller for testing
        dropout=0.1
    )


@pytest.fixture
def sample_model(sample_model_config):
    """Create sample model for testing."""
    return GenreAdaptiveNLIValidator(sample_model_config)


@pytest.fixture
def sample_nli_data():
    """Create sample NLI data for testing."""
    return [
        {
            "premise": "The cat is sleeping on the couch.",
            "hypothesis": "A cat is resting on furniture.",
            "label": 0,  # entailment
            "genre": "fiction"
        },
        {
            "premise": "Scientists discovered a new species of bird.",
            "hypothesis": "A new bird was found by researchers.",
            "label": 0,  # entailment
            "genre": "news"
        },
        {
            "premise": "The weather is sunny today.",
            "hypothesis": "It's raining heavily outside.",
            "label": 2,  # contradiction
            "genre": "news"
        },
        {
            "premise": "The book contains 300 pages.",
            "hypothesis": "The novel has many chapters.",
            "label": 1,  # neutral
            "genre": "fiction"
        }
    ]


@pytest.fixture
def sample_summarization_data():
    """Create sample summarization data for testing."""
    return [
        {
            "article": "Scientists at the university have made a breakthrough discovery in quantum computing. "
                      "The new technique allows for faster processing of complex algorithms. "
                      "This could revolutionize how we approach computational problems.",
            "highlights": "Scientists made a quantum computing breakthrough that enables faster algorithm processing.",
            "genre": "news"
        },
        {
            "article": "The protagonist walked through the dark forest, hearing mysterious sounds from the shadows. "
                      "Every step seemed to echo through the trees, creating an atmosphere of suspense and fear. "
                      "She knew she had to find the hidden treasure before dawn.",
            "highlights": "A character searches for treasure in a dark, mysterious forest at night.",
            "genre": "fiction"
        }
    ]


@pytest.fixture
def sample_dataset(sample_nli_data):
    """Create sample dataset for testing."""
    return Dataset.from_list(sample_nli_data)


@pytest.fixture
def sample_tokenized_data(sample_nli_data, sample_tokenizer):
    """Create sample tokenized data for testing."""
    tokenized_data = []

    for example in sample_nli_data:
        # Tokenize premise and hypothesis
        inputs = sample_tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors=None
        )

        # Add labels and genre
        inputs["labels"] = example["label"]
        inputs["genre"] = example["genre"]

        tokenized_data.append(inputs)

    return Dataset.from_list(tokenized_data)


@pytest.fixture
def data_loader():
    """Create sample data loader for testing."""
    return SummarizationDataLoader(
        tokenizer_name="microsoft/deberta-v3-base",
        seed=42
    )


@pytest.fixture
def sample_predictions():
    """Create sample predictions for metrics testing."""
    return {
        "predictions": [0, 1, 2, 0, 1],
        "labels": [0, 1, 2, 1, 0],
        "probabilities": [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.4, 0.4, 0.2]
        ],
        "genres": ["fiction", "news", "academic", "fiction", "news"]
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    import random
    import numpy as np

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

    # Ensure deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_mlflow_run():
    """Mock MLflow run for testing."""
    import unittest.mock

    with unittest.mock.patch('mlflow.start_run'):
        with unittest.mock.patch('mlflow.log_param'):
            with unittest.mock.patch('mlflow.log_metric'):
                with unittest.mock.patch('mlflow.log_artifact'):
                    yield


# Device fixtures for testing on different hardware
@pytest.fixture(params=["cpu"])  # Add "cuda" if GPU testing is needed
def device(request):
    """Test device fixture."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)


# Parameterized fixtures for testing different configurations
@pytest.fixture(params=[1, 2])
def genre_adaptation_layers(request):
    """Test different numbers of genre adaptation layers."""
    return request.param


@pytest.fixture(params=[32, 64])
def genre_embedding_dim(request):
    """Test different genre embedding dimensions."""
    return request.param


@pytest.fixture(params=["fiction", "news", "academic", "government"])
def sample_genre(request):
    """Test different genres."""
    return request.param


# Helper functions for tests
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert that tensor has expected shape."""
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, (
        f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor"):
    """Assert that tensor has expected dtype."""
    assert tensor.dtype == expected_dtype, (
        f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
    )


def create_mock_batch(batch_size: int = 2, seq_len: int = 128, num_labels: int = 3):
    """Create mock batch for testing."""
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "token_type_ids": torch.zeros(batch_size, seq_len),
        "genre_ids": torch.randint(0, 5, (batch_size,)),
        "labels": torch.randint(0, num_labels, (batch_size,))
    }


# Performance benchmarking helpers
@pytest.fixture
def benchmark_timer():
    """Simple timer for performance benchmarking."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.elapsed()

        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

    return Timer()