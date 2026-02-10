"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset

from genre_adaptive_nli_summarization_validator.data.loader import SummarizationDataLoader
from genre_adaptive_nli_summarization_validator.data.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def test_init(self, sample_text_preprocessor):
        """Test TextPreprocessor initialization."""
        preprocessor = sample_text_preprocessor
        assert preprocessor.tokenizer is not None
        assert len(preprocessor.stop_words) > 0

    def test_clean_text(self, sample_text_preprocessor):
        """Test text cleaning functionality."""
        preprocessor = sample_text_preprocessor

        # Test basic cleaning
        dirty_text = "  This   is    a    test!!!   "
        clean_text = preprocessor.clean_text(dirty_text)
        assert clean_text == "This is a test!"

        # Test empty input
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

        # Test special characters
        special_text = "Hello@#$%world!!!"
        clean_special = preprocessor.clean_text(special_text)
        assert "@#$%" not in clean_special
        assert "Hello" in clean_special and "world" in clean_special

    def test_truncate_text(self, sample_text_preprocessor):
        """Test text truncation functionality."""
        preprocessor = sample_text_preprocessor

        # Test short text (no truncation needed)
        short_text = "This is a short text."
        truncated = preprocessor.truncate_text(short_text, max_length=100)
        assert truncated == short_text

        # Test long text (truncation needed)
        long_text = " ".join(["word"] * 200)
        truncated = preprocessor.truncate_text(long_text, max_length=50)
        assert len(preprocessor.tokenizer.encode(truncated)) <= 50

        # Test empty text
        assert preprocessor.truncate_text("", max_length=100) == ""

    def test_extract_genre_indicators(self, sample_text_preprocessor):
        """Test genre indicator extraction."""
        preprocessor = sample_text_preprocessor

        # Test fiction indicators
        fiction_text = "The character in this novel had an interesting dialogue with the protagonist."
        indicators = preprocessor.extract_genre_indicators(fiction_text)
        assert "fiction" in indicators
        assert indicators["fiction"] > 0

        # Test news indicators
        news_text = "According to official sources, the press reported breaking news today."
        indicators = preprocessor.extract_genre_indicators(news_text)
        assert "news" in indicators
        assert indicators["news"] > 0

        # Test empty text
        empty_indicators = preprocessor.extract_genre_indicators("")
        assert isinstance(empty_indicators, dict)

    def test_create_nli_pairs(self, sample_text_preprocessor):
        """Test NLI pair creation."""
        preprocessor = sample_text_preprocessor

        document = "Scientists discovered a new species of bird in the Amazon rainforest."
        summary = "A new bird species was found in the Amazon."

        pairs = preprocessor.create_nli_pairs(document, summary)

        assert isinstance(pairs, list)
        assert len(pairs) > 0

        # Check pair structure
        for premise, hypothesis, label in pairs:
            assert isinstance(premise, str)
            assert isinstance(hypothesis, str)
            assert label in ["entailment", "neutral", "contradiction"]

    def test_validate_summary_quality(self, sample_text_preprocessor):
        """Test summary quality validation."""
        preprocessor = sample_text_preprocessor

        document = "The research team published their findings in a peer-reviewed journal. " \
                  "The study involved 1000 participants over a period of two years."
        summary = "Researchers published study results with 1000 participants."

        metrics = preprocessor.validate_summary_quality(document, summary)

        assert isinstance(metrics, dict)
        assert "length_ratio" in metrics
        assert "vocabulary_coverage" in metrics
        assert "abstractiveness" in metrics
        assert "sentence_length_ratio" in metrics

        # Check metric ranges
        assert 0 <= metrics["length_ratio"] <= 1
        assert 0 <= metrics["vocabulary_coverage"] <= 1
        assert 0 <= metrics["abstractiveness"] <= 1

    def test_prepare_model_inputs(self, sample_text_preprocessor):
        """Test model input preparation."""
        preprocessor = sample_text_preprocessor

        premise = "The cat is sleeping on the couch."
        hypothesis = "A cat is resting."

        inputs = preprocessor.prepare_model_inputs(premise, hypothesis, max_length=128)

        # Check required keys
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        # Check shapes
        assert len(inputs["input_ids"]) == 128
        assert len(inputs["attention_mask"]) == 128

        # Check types
        assert isinstance(inputs["input_ids"][0], int)
        assert isinstance(inputs["attention_mask"][0], int)


class TestSummarizationDataLoader:
    """Test cases for SummarizationDataLoader class."""

    def test_init(self, data_loader):
        """Test SummarizationDataLoader initialization."""
        assert data_loader.tokenizer is not None
        assert data_loader.preprocessor is not None
        assert isinstance(data_loader.genre_mappings, dict)

    @pytest.mark.slow
    def test_load_multinli_dataset(self, data_loader):
        """Test MultiNLI dataset loading."""
        # Skip if network is not available
        try:
            dataset = data_loader.load_multinli_dataset(
                split="validation_matched",
                max_samples=10
            )
            assert len(dataset) <= 10
            assert "premise" in dataset.column_names
            assert "hypothesis" in dataset.column_names
            assert "genre" in dataset.column_names
        except Exception as e:
            pytest.skip(f"Network dataset loading failed: {e}")

    @pytest.mark.slow
    def test_load_cnn_dailymail_dataset(self, data_loader):
        """Test CNN/DailyMail dataset loading."""
        try:
            dataset = data_loader.load_cnn_dailymail_dataset(
                split="validation",
                max_samples=5
            )
            assert len(dataset) <= 5
            assert "article" in dataset.column_names
            assert "highlights" in dataset.column_names
            assert "genre" in dataset.column_names
        except Exception as e:
            pytest.skip(f"Network dataset loading failed: {e}")

    def test_create_nli_training_data(self, data_loader, sample_summarization_data):
        """Test NLI training data creation."""
        # Create dataset from sample data
        summarization_dataset = Dataset.from_list(sample_summarization_data)

        nli_dataset = data_loader.create_nli_training_data(
            summarization_dataset,
            negative_sampling_ratio=0.5,
            max_length=128
        )

        # Check dataset structure
        assert len(nli_dataset) > 0
        assert "input_ids" in nli_dataset.column_names
        assert "attention_mask" in nli_dataset.column_names
        assert "labels" in nli_dataset.column_names
        assert "genre" in nli_dataset.column_names

        # Check label distribution
        labels = nli_dataset["labels"]
        unique_labels = set(labels)
        assert 0 in unique_labels  # entailment
        assert len(unique_labels) > 1  # should have multiple labels

    def test_create_genre_balanced_split(self, data_loader, sample_nli_data):
        """Test genre-balanced dataset splitting."""
        dataset = Dataset.from_list(sample_nli_data)

        splits = data_loader.create_genre_balanced_split(
            dataset,
            test_size=0.3,
            val_size=0.2
        )

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        total_samples = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total_samples == len(dataset)

    def test_get_genre_statistics(self, data_loader, sample_nli_data):
        """Test genre statistics calculation."""
        dataset = Dataset.from_list(sample_nli_data)
        stats = data_loader.get_genre_statistics(dataset)

        assert isinstance(stats, dict)
        assert "fiction" in stats
        assert "news" in stats
        assert all(isinstance(count, int) for count in stats.values())

    def test_normalize_genre(self, data_loader):
        """Test genre normalization."""
        # Test known mappings
        assert data_loader._normalize_genre("fiction") == "fiction"
        assert data_loader._normalize_genre("FICTION") == "fiction"

        # Test unknown genre
        unknown_genre = data_loader._normalize_genre("unknown_genre")
        assert unknown_genre == "unknown_genre"

    def test_label_to_int(self, data_loader):
        """Test label to integer conversion."""
        assert data_loader._label_to_int("entailment") == 0
        assert data_loader._label_to_int("neutral") == 1
        assert data_loader._label_to_int("contradiction") == 2
        assert data_loader._label_to_int("ENTAILMENT") == 0

    def test_load_custom_dataset(self, data_loader, temp_dir):
        """Test custom dataset loading."""
        # Create sample CSV file
        csv_data = pd.DataFrame({
            "text": ["Document 1", "Document 2"],
            "summary": ["Summary 1", "Summary 2"],
            "genre": ["fiction", "news"]
        })

        csv_path = temp_dir / "test_data.csv"
        csv_data.to_csv(csv_path, index=False)

        # Load dataset
        dataset = data_loader.load_custom_dataset(
            csv_path,
            text_column="text",
            summary_column="summary",
            genre_column="genre"
        )

        assert len(dataset) == 2
        assert "article" in dataset.column_names  # renamed from text
        assert "highlights" in dataset.column_names  # renamed from summary
        assert "genre" in dataset.column_names

    def test_load_custom_dataset_missing_file(self, data_loader, temp_dir):
        """Test custom dataset loading with missing file."""
        non_existent_path = temp_dir / "non_existent.csv"

        with pytest.raises(FileNotFoundError):
            data_loader.load_custom_dataset(non_existent_path)

    def test_load_custom_dataset_missing_columns(self, data_loader, temp_dir):
        """Test custom dataset loading with missing required columns."""
        # Create CSV with missing columns
        csv_data = pd.DataFrame({
            "wrong_column": ["data1", "data2"]
        })

        csv_path = temp_dir / "bad_data.csv"
        csv_data.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            data_loader.load_custom_dataset(csv_path)


class TestDataIntegration:
    """Integration tests for data loading and preprocessing."""

    def test_end_to_end_data_pipeline(self, data_loader, sample_summarization_data):
        """Test complete data pipeline from raw data to model inputs."""
        # Create dataset
        raw_dataset = Dataset.from_list(sample_summarization_data)

        # Create NLI training data
        nli_dataset = data_loader.create_nli_training_data(
            raw_dataset,
            negative_sampling_ratio=0.3,
            max_length=128
        )

        # Verify pipeline output
        assert len(nli_dataset) > 0

        # Check first example
        example = nli_dataset[0]
        assert isinstance(example["input_ids"], list)
        assert isinstance(example["attention_mask"], list)
        assert isinstance(example["labels"], int)
        assert isinstance(example["genre"], str)

        # Check tensor conversion works
        input_ids = torch.tensor(example["input_ids"])
        attention_mask = torch.tensor(example["attention_mask"])
        labels = torch.tensor(example["labels"])

        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long
        assert labels.dtype == torch.long

    def test_genre_consistency(self, data_loader, sample_nli_data):
        """Test genre consistency across data processing steps."""
        dataset = Dataset.from_list(sample_nli_data)

        # Get original genres
        original_genres = set(example["genre"] for example in sample_nli_data)

        # Process through pipeline
        genre_stats = data_loader.get_genre_statistics(dataset)

        # Check all original genres are preserved
        for genre in original_genres:
            assert genre in genre_stats
            assert genre_stats[genre] > 0

    def test_data_preprocessing_edge_cases(self, sample_text_preprocessor):
        """Test preprocessing with edge cases."""
        preprocessor = sample_text_preprocessor

        # Empty inputs
        assert preprocessor.clean_text("") == ""
        assert preprocessor.truncate_text("") == ""

        # Very long inputs
        very_long_text = "word " * 1000
        truncated = preprocessor.truncate_text(very_long_text, max_length=50)
        tokens = preprocessor.tokenizer.encode(truncated)
        assert len(tokens) <= 50

        # Special characters and unicode
        unicode_text = "Hello 世界 🌍 test!"
        cleaned = preprocessor.clean_text(unicode_text)
        assert isinstance(cleaned, str)

    def test_batch_processing_performance(self, data_loader, sample_nli_data, benchmark_timer):
        """Test batch processing performance."""
        # Create larger dataset for performance testing
        large_data = sample_nli_data * 25  # 100 samples
        dataset = Dataset.from_list(large_data)

        benchmark_timer.start()
        genre_stats = data_loader.get_genre_statistics(dataset)
        elapsed = benchmark_timer.stop()

        # Should process 100 samples quickly
        assert elapsed < 1.0  # Should complete in less than 1 second
        assert len(genre_stats) > 0