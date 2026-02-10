"""Data loading utilities for multi-genre summarization datasets."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import numpy as np

from .preprocessing import TextPreprocessor
from ..exceptions import DataLoadError


class SummarizationDataLoader:
    """Data loader for multi-genre summarization and NLI datasets."""

    def __init__(
        self,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize data loader.

        Args:
            tokenizer_name: HuggingFace tokenizer name.
            cache_dir: Directory to cache datasets.
            seed: Random seed for reproducibility.
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing SummarizationDataLoader with tokenizer: {tokenizer_name}")
        self.logger.debug(f"Cache directory: {cache_dir}, Seed: {seed}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.preprocessor = TextPreprocessor(tokenizer_name)
        self.cache_dir = cache_dir
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        self.logger.info("Data loader initialized successfully")

        # Genre mappings
        self.genre_mappings = {
            'fiction': ['fiction', 'fictionbooks'],
            'government': ['government'],
            'slate': ['slate'],
            'telephone': ['telephone'],
            'travel': ['travel'],
            'letters': ['letters'],
            'oup': ['oup'],
            'nineeleven': ['9/11report'],
            'face-to-face': ['face-to-face'],
            'verbatim': ['verbatim']
        }

    def load_multinli_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Load MultiNLI dataset with genre information.

        Args:
            split: Dataset split ('train', 'validation_matched', 'validation_mismatched').
            max_samples: Maximum number of samples to load.

        Returns:
            Loaded and processed dataset.
        """
        self.logger.info(f"Loading MultiNLI {split} dataset...")

        try:
            dataset = load_dataset("multi_nli", cache_dir=self.cache_dir)[split]
        except Exception as e:
            self.logger.error(f"Failed to load MultiNLI dataset: {e}")
            raise DataLoadError(
                f"Failed to load MultiNLI {split} dataset",
                details={"split": split, "cache_dir": self.cache_dir},
                original_exception=e
            )

        # Add genre labels
        dataset = dataset.map(
            lambda example: {"genre": self._normalize_genre(example["genre"])},
            desc="Adding genre labels",
            batched=False,
            load_from_cache_file=True
        )

        # Limit samples if specified
        if max_samples is not None and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        self.logger.info(f"Loaded {len(dataset)} samples from MultiNLI {split}")
        return dataset

    def load_cnn_dailymail_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        version: str = "3.0.0"
    ) -> Dataset:
        """Load CNN/DailyMail summarization dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            max_samples: Maximum number of samples to load.
            version: Dataset version.

        Returns:
            Loaded and processed dataset.
        """
        self.logger.info(f"Loading CNN/DailyMail {split} dataset...")

        try:
            dataset = load_dataset("cnn_dailymail", version, cache_dir=self.cache_dir)[split]
        except Exception as e:
            self.logger.error(f"Failed to load CNN/DailyMail dataset: {e}")
            raise

        # Add genre information (news domain)
        dataset = dataset.map(
            lambda example: {"genre": "news"},
            desc="Adding genre labels",
            batched=False,
            load_from_cache_file=True
        )

        # Limit samples if specified
        if max_samples is not None and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        self.logger.info(f"Loaded {len(dataset)} samples from CNN/DailyMail {split}")
        return dataset

    def load_xsum_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Load XSum dataset for additional summarization data.

        Args:
            split: Dataset split ('train', 'validation', 'test').
            max_samples: Maximum number of samples to load.

        Returns:
            Loaded and processed dataset.
        """
        self.logger.info(f"Loading XSum {split} dataset...")

        try:
            dataset = load_dataset("xsum", cache_dir=self.cache_dir)[split]
        except Exception as e:
            self.logger.error(f"Failed to load XSum dataset: {e}")
            raise

        # Add genre information (news domain)
        dataset = dataset.map(
            lambda example: {"genre": "news"},
            desc="Adding genre labels",
            batched=False,
            load_from_cache_file=True
        )

        # Rename columns to match our schema
        dataset = dataset.map(
            lambda example: {
                "article": example["document"],
                "highlights": example["summary"],
                "genre": example["genre"]
            },
            remove_columns=["document", "summary"]
        )

        # Limit samples if specified
        if max_samples is not None and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        self.logger.info(f"Loaded {len(dataset)} samples from XSum {split}")
        return dataset

    def create_nli_training_data(
        self,
        summarization_dataset: Dataset,
        negative_sampling_ratio: float = 0.3,
        max_length: int = 512
    ) -> Dataset:
        """Create NLI training data from summarization datasets.

        Args:
            summarization_dataset: Source summarization dataset.
            negative_sampling_ratio: Ratio of negative samples to create.
            max_length: Maximum sequence length.

        Returns:
            NLI training dataset.
        """
        self.logger.info("Creating NLI training data from summarization dataset...")

        nli_examples = []

        for example in summarization_dataset:
            article = example.get("article", "")
            highlights = example.get("highlights", "")
            genre = example.get("genre", "unknown")

            if not article or not highlights:
                continue

            # Create entailment pairs
            nli_pairs = self.preprocessor.create_nli_pairs(article, highlights)

            for premise, hypothesis, label in nli_pairs:
                nli_examples.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": self._label_to_int(label),
                    "genre": genre
                })

        # Add negative samples (contradictions)
        negative_count = int(len(nli_examples) * negative_sampling_ratio)
        negative_examples = self._create_negative_samples(
            nli_examples, negative_count
        )
        nli_examples.extend(negative_examples)

        # Shuffle examples
        random.shuffle(nli_examples)

        # Convert to dataset and tokenize
        dataset = Dataset.from_list(nli_examples)
        dataset = dataset.map(
            lambda example: self._tokenize_nli_example(example, max_length),
            desc="Tokenizing NLI examples",
            batched=False,
            load_from_cache_file=True
        )

        self.logger.info(f"Created {len(dataset)} NLI training examples")
        return dataset

    def _create_negative_samples(
        self,
        positive_examples: List[Dict[str, Any]],
        negative_count: int
    ) -> List[Dict[str, Any]]:
        """Create negative (contradiction) samples by hypothesis shuffling.

        Args:
            positive_examples: List of positive examples.
            negative_count: Number of negative samples to create.

        Returns:
            List of negative examples.
        """
        negative_examples = []
        premises = [ex["premise"] for ex in positive_examples]
        hypotheses = [ex["hypothesis"] for ex in positive_examples]
        genres = [ex["genre"] for ex in positive_examples]

        for _ in range(negative_count):
            # Random mismatched premise-hypothesis pair
            prem_idx = random.randint(0, len(premises) - 1)
            hyp_idx = random.randint(0, len(hypotheses) - 1)

            # Ensure they're actually different
            if prem_idx != hyp_idx:
                negative_examples.append({
                    "premise": premises[prem_idx],
                    "hypothesis": hypotheses[hyp_idx],
                    "label": self._label_to_int("contradiction"),
                    "genre": genres[prem_idx]
                })

        return negative_examples

    def _tokenize_nli_example(
        self,
        example: Dict[str, Any],
        max_length: int
    ) -> Dict[str, Any]:
        """Tokenize NLI example for model input.

        Args:
            example: Raw example dictionary.
            max_length: Maximum sequence length.

        Returns:
            Tokenized example.
        """
        inputs = self.preprocessor.prepare_model_inputs(
            example["premise"],
            example["hypothesis"],
            max_length
        )

        return {
            **inputs,
            "labels": example["label"],
            "genre": example["genre"]
        }

    def create_genre_balanced_split(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> DatasetDict:
        """Create genre-balanced train/validation/test splits.

        Args:
            dataset: Dataset to split.
            test_size: Proportion for test set.
            val_size: Proportion for validation set.

        Returns:
            DatasetDict with train/validation/test splits.
        """
        self.logger.info("Creating genre-balanced dataset splits...")

        # Group by genre
        genre_groups = {}
        for i, example in enumerate(dataset):
            genre = example["genre"]
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append(i)

        train_indices, val_indices, test_indices = [], [], []

        # Split each genre separately
        for genre, indices in genre_groups.items():
            random.shuffle(indices)
            n_samples = len(indices)

            n_test = int(n_samples * test_size)
            n_val = int(n_samples * val_size)
            n_train = n_samples - n_test - n_val

            test_indices.extend(indices[:n_test])
            val_indices.extend(indices[n_test:n_test + n_val])
            train_indices.extend(indices[n_test + n_val:])

        # Create splits
        splits = DatasetDict({
            "train": dataset.select(train_indices),
            "validation": dataset.select(val_indices),
            "test": dataset.select(test_indices)
        })

        self.logger.info(
            f"Created splits - Train: {len(splits['train'])}, "
            f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}"
        )

        return splits

    def get_genre_statistics(self, dataset: Dataset) -> Dict[str, int]:
        """Get genre distribution statistics.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Dictionary of genre counts.
        """
        genre_counts = {}
        for example in dataset:
            genre = example.get("genre", "unknown")
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        return dict(sorted(genre_counts.items()))

    def _normalize_genre(self, genre: str) -> str:
        """Normalize genre labels to standard categories.

        Args:
            genre: Original genre label.

        Returns:
            Normalized genre label.
        """
        genre = genre.lower().strip()

        for normalized, variants in self.genre_mappings.items():
            if genre in variants:
                return normalized

        # Default mapping for unknown genres
        return genre

    def _label_to_int(self, label: str) -> int:
        """Convert string labels to integer labels.

        Args:
            label: String label.

        Returns:
            Integer label (0=entailment, 1=neutral, 2=contradiction).
        """
        label_map = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
        return label_map.get(label.lower(), 1)

    def load_custom_dataset(
        self,
        data_path: Union[str, Path],
        text_column: str = "text",
        summary_column: str = "summary",
        genre_column: str = "genre"
    ) -> Dataset:
        """Load custom dataset from CSV/JSON file.

        Args:
            data_path: Path to data file.
            text_column: Name of text column.
            summary_column: Name of summary column.
            genre_column: Name of genre column.

        Returns:
            Loaded dataset.
        """
        data_path = Path(data_path)
        self.logger.info(f"Loading custom dataset from {data_path}")

        if not data_path.exists():
            raise DataLoadError(
                "Dataset file not found",
                details={"data_path": str(data_path)}
            )

        try:
            # Load based on file extension
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() == '.json':
                df = pd.read_json(data_path, lines=True)
            else:
                raise DataLoadError(
                    "Unsupported file format",
                    details={"file_format": data_path.suffix, "data_path": str(data_path)}
                )
        except (pd.errors.ParserError, ValueError) as e:
            raise DataLoadError(
                "Failed to parse dataset file",
                details={"data_path": str(data_path), "file_format": data_path.suffix},
                original_exception=e
            )

        # Validate required columns
        required_cols = [text_column, summary_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataLoadError(
                "Missing required columns in dataset",
                details={
                    "missing_columns": missing_cols,
                    "available_columns": list(df.columns),
                    "data_path": str(data_path)
                }
            )

        # Add genre column if missing
        if genre_column not in df.columns:
            df[genre_column] = "unknown"

        # Rename columns to standard format
        df = df.rename(columns={
            text_column: "article",
            summary_column: "highlights",
            genre_column: "genre"
        })

        # Convert to dataset
        dataset = Dataset.from_pandas(df)

        self.logger.info(f"Loaded {len(dataset)} samples from custom dataset")
        return dataset