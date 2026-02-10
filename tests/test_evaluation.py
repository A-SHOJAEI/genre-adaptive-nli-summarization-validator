"""Tests for evaluation metrics and analysis tools."""

import pytest
import numpy as np
from pathlib import Path

from genre_adaptive_nli_summarization_validator.evaluation.metrics import SummaryValidationMetrics


class TestSummaryValidationMetrics:
    """Test cases for SummaryValidationMetrics class."""

    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator instance."""
        return SummaryValidationMetrics()

    def test_init(self, metrics_calculator):
        """Test metrics calculator initialization."""
        assert metrics_calculator.label_names == ["entailment", "neutral", "contradiction"]

    def test_compute_basic_metrics(self, metrics_calculator, sample_predictions):
        """Test basic classification metrics computation."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        probabilities = sample_predictions["probabilities"]

        metrics = metrics_calculator.compute_basic_metrics(predictions, labels, probabilities)

        # Check required metrics exist
        required_metrics = [
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_micro", "recall_micro", "f1_micro"
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

        # Check class-specific metrics
        for label_name in metrics_calculator.label_names:
            assert f"precision_{label_name}" in metrics
            assert f"recall_{label_name}" in metrics
            assert f"f1_{label_name}" in metrics

        # Check AUC metrics
        assert "entailment_auc" in metrics
        assert "entailment_ap" in metrics
        assert 0 <= metrics["entailment_auc"] <= 1

    def test_compute_basic_metrics_without_probabilities(self, metrics_calculator, sample_predictions):
        """Test basic metrics computation without probabilities."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]

        metrics = metrics_calculator.compute_basic_metrics(predictions, labels)

        # Should have basic metrics but not AUC metrics
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "entailment_auc" not in metrics

    def test_compute_genre_specific_metrics(self, metrics_calculator, sample_predictions):
        """Test genre-specific metrics computation."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        genres = sample_predictions["genres"]
        probabilities = sample_predictions["probabilities"]

        genre_metrics = metrics_calculator.compute_genre_specific_metrics(
            predictions, labels, genres, probabilities
        )

        # Check structure
        assert isinstance(genre_metrics, dict)
        assert len(genre_metrics) > 0

        # Check each genre has metrics
        for genre, metrics in genre_metrics.items():
            assert isinstance(metrics, dict)
            assert "accuracy" in metrics
            assert "sample_count" in metrics
            assert metrics["sample_count"] > 0

    def test_compute_hallucination_detection_metrics(self, metrics_calculator, sample_predictions):
        """Test hallucination detection metrics computation."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        probabilities = sample_predictions["probabilities"]

        halluc_metrics = metrics_calculator.compute_hallucination_detection_metrics(
            predictions, labels, probabilities
        )

        # Check required metrics
        required_metrics = [
            "hallucination_detection_accuracy",
            "hallucination_detection_precision",
            "hallucination_detection_recall",
            "hallucination_detection_f1"
        ]
        for metric in required_metrics:
            assert metric in halluc_metrics
            assert 0 <= halluc_metrics[metric] <= 1

    def test_compute_calibration_metrics(self, metrics_calculator, sample_predictions):
        """Test calibration metrics computation."""
        predictions = sample_predictions["predictions"]
        probabilities = sample_predictions["probabilities"]

        calib_metrics = metrics_calculator.compute_calibration_metrics(predictions, probabilities)

        # Check required metrics
        assert "expected_calibration_error" in calib_metrics
        assert "maximum_calibration_error" in calib_metrics
        assert "reliability_diagram_data" in calib_metrics

        # Check ECE and MCE are reasonable
        assert 0 <= calib_metrics["expected_calibration_error"] <= 1
        assert 0 <= calib_metrics["maximum_calibration_error"] <= 1

        # Check reliability diagram data structure
        diagram_data = calib_metrics["reliability_diagram_data"]
        assert "fraction_of_positives" in diagram_data
        assert "mean_predicted_value" in diagram_data

    def test_compute_genre_transfer_metrics(self, metrics_calculator):
        """Test genre transfer metrics computation."""
        source_metrics = {"accuracy": 0.85, "f1_macro": 0.80}
        target_metrics = {
            "fiction": {"accuracy": 0.82, "f1_macro": 0.78},
            "news": {"accuracy": 0.88, "f1_macro": 0.85},
            "academic": {"accuracy": 0.80, "f1_macro": 0.76}
        }

        transfer_metrics = metrics_calculator.compute_genre_transfer_metrics(
            source_metrics, target_metrics
        )

        # Check required metrics
        assert "genre_accuracy_mean" in transfer_metrics
        assert "genre_accuracy_std" in transfer_metrics
        assert "genre_accuracy_cv" in transfer_metrics
        assert "genre_f1_consistency" in transfer_metrics

        # Check value ranges
        assert transfer_metrics["genre_accuracy_mean"] > 0
        assert transfer_metrics["genre_accuracy_std"] >= 0

    def test_compute_genre_transfer_metrics_with_baseline(self, metrics_calculator):
        """Test genre transfer metrics with baseline comparison."""
        source_metrics = {"accuracy": 0.85}
        target_metrics = {
            "fiction": {"accuracy": 0.82},
            "news": {"accuracy": 0.88}
        }
        baseline_metrics = {
            "fiction": {"accuracy": 0.75},
            "news": {"accuracy": 0.80}
        }

        transfer_metrics = metrics_calculator.compute_genre_transfer_metrics(
            source_metrics, target_metrics, baseline_metrics
        )

        # Should have improvement metrics
        assert "mean_improvement" in transfer_metrics
        assert "improvement_consistency" in transfer_metrics

    def test_create_evaluation_report(self, metrics_calculator, sample_predictions, temp_dir):
        """Test comprehensive evaluation report creation."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        genres = sample_predictions["genres"]
        probabilities = sample_predictions["probabilities"]

        report = metrics_calculator.create_evaluation_report(
            predictions, labels, genres, probabilities
        )

        # Check report structure
        required_sections = [
            "overall_metrics", "genre_metrics", "hallucination_metrics",
            "calibration_metrics", "transfer_metrics", "confusion_matrix",
            "classification_report", "genre_distribution", "sample_statistics"
        ]
        for section in required_sections:
            assert section in report

        # Check sample statistics
        sample_stats = report["sample_statistics"]
        assert sample_stats["total_samples"] == len(predictions)
        assert sample_stats["num_genres"] > 0

    def test_create_evaluation_report_with_save(self, metrics_calculator, sample_predictions, temp_dir):
        """Test evaluation report creation with file saving."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        genres = sample_predictions["genres"]
        probabilities = sample_predictions["probabilities"]

        save_path = temp_dir / "evaluation_report.json"

        report = metrics_calculator.create_evaluation_report(
            predictions, labels, genres, probabilities, str(save_path)
        )

        # Check file was created
        assert save_path.exists()

        # Check file contents
        import json
        with open(save_path, "r") as f:
            saved_report = json.load(f)

        assert saved_report == report

    def test_compute_target_metrics(self, metrics_calculator, sample_predictions):
        """Test target metrics computation."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]
        genres = sample_predictions["genres"]
        probabilities = sample_predictions["probabilities"]

        target_metrics = metrics_calculator.compute_target_metrics(
            predictions, labels, genres, probabilities
        )

        # Check target metrics from project requirements
        expected_metrics = [
            "entailment_auc",
            "hallucination_detection_f1",
            "genre_transfer_accuracy"
        ]
        for metric in expected_metrics:
            assert metric in target_metrics
            assert 0 <= target_metrics[metric] <= 1

    def test_plot_confusion_matrix(self, metrics_calculator, sample_predictions, temp_dir):
        """Test confusion matrix plotting."""
        predictions = sample_predictions["predictions"]
        labels = sample_predictions["labels"]

        save_path = temp_dir / "confusion_matrix.png"

        # Should not raise an exception
        try:
            metrics_calculator.plot_confusion_matrix(
                predictions, labels, str(save_path), normalize=True
            )
            # Check file was created
            assert save_path.exists()
        except ImportError:
            # Skip if matplotlib not available in test environment
            pytest.skip("Matplotlib not available for plotting tests")

    def test_plot_genre_performance(self, metrics_calculator, temp_dir):
        """Test genre performance plotting."""
        genre_metrics = {
            "fiction": {"accuracy": 0.82},
            "news": {"accuracy": 0.88},
            "academic": {"accuracy": 0.80}
        }

        save_path = temp_dir / "genre_performance.png"

        try:
            metrics_calculator.plot_genre_performance(
                genre_metrics, "accuracy", str(save_path)
            )
            assert save_path.exists()
        except ImportError:
            pytest.skip("Matplotlib not available for plotting tests")

    def test_edge_cases(self, metrics_calculator):
        """Test edge cases and error handling."""
        # Empty inputs
        empty_metrics = metrics_calculator.compute_basic_metrics([], [])
        assert isinstance(empty_metrics, dict)

        # Single class
        single_class_metrics = metrics_calculator.compute_basic_metrics([0, 0, 0], [0, 0, 0])
        assert isinstance(single_class_metrics, dict)

        # Mismatched lengths should be handled gracefully
        try:
            metrics_calculator.compute_basic_metrics([0, 1], [0])
        except (ValueError, IndexError):
            pass  # Expected to fail

    def test_perfect_predictions(self, metrics_calculator):
        """Test metrics with perfect predictions."""
        predictions = [0, 1, 2, 0, 1]
        labels = [0, 1, 2, 0, 1]  # Perfect match
        probabilities = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        genres = ["fiction", "news", "academic", "fiction", "news"]

        metrics = metrics_calculator.compute_basic_metrics(predictions, labels, probabilities)

        # Perfect predictions should yield accuracy = 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_random_predictions(self, metrics_calculator):
        """Test metrics with random predictions."""
        np.random.seed(42)

        n_samples = 100
        n_classes = 3

        # Generate random predictions and labels
        predictions = np.random.randint(0, n_classes, n_samples).tolist()
        labels = np.random.randint(0, n_classes, n_samples).tolist()

        # Generate random probabilities that sum to 1
        probabilities = np.random.dirichlet(np.ones(n_classes), n_samples).tolist()

        genres = np.random.choice(["fiction", "news", "academic"], n_samples).tolist()

        metrics = metrics_calculator.compute_basic_metrics(predictions, labels, probabilities)

        # Random predictions should yield reasonable metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_class_imbalance(self, metrics_calculator):
        """Test metrics with class imbalanced data."""
        # Heavily imbalanced towards class 0
        predictions = [0] * 90 + [1] * 5 + [2] * 5
        labels = [0] * 90 + [1] * 5 + [2] * 5
        probabilities = [[0.9, 0.05, 0.05]] * 90 + [[0.1, 0.8, 0.1]] * 5 + [[0.1, 0.1, 0.8]] * 5
        genres = ["fiction"] * 100

        metrics = metrics_calculator.compute_basic_metrics(predictions, labels, probabilities)

        # Micro and macro averages should differ with class imbalance
        assert metrics["f1_micro"] != metrics["f1_macro"]

    def test_genre_specific_edge_cases(self, metrics_calculator):
        """Test genre-specific metrics with edge cases."""
        predictions = [0, 1, 2]
        labels = [0, 1, 2]
        genres = ["fiction", "fiction", "fiction"]  # Single genre
        probabilities = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        genre_metrics = metrics_calculator.compute_genre_specific_metrics(
            predictions, labels, genres, probabilities
        )

        # Should have metrics for fiction genre
        assert "fiction" in genre_metrics
        assert genre_metrics["fiction"]["accuracy"] == 1.0

    def test_calibration_edge_cases(self, metrics_calculator):
        """Test calibration metrics with edge cases."""
        # Perfect calibration case
        predictions = [0, 1, 0, 1]
        probabilities = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]

        # Expand to 3 classes for consistency
        probabilities_3class = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05],
                               [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]

        calib_metrics = metrics_calculator.compute_calibration_metrics(predictions, probabilities_3class)

        assert isinstance(calib_metrics, dict)
        assert "expected_calibration_error" in calib_metrics

    def test_memory_efficiency_large_dataset(self, metrics_calculator):
        """Test memory efficiency with larger dataset."""
        n_samples = 10000
        np.random.seed(42)

        predictions = np.random.randint(0, 3, n_samples).tolist()
        labels = np.random.randint(0, 3, n_samples).tolist()
        probabilities = np.random.dirichlet(np.ones(3), n_samples).tolist()
        genres = np.random.choice(["fiction", "news", "academic"], n_samples).tolist()

        # Should handle large datasets without memory issues
        metrics = metrics_calculator.compute_basic_metrics(predictions, labels, probabilities)
        genre_metrics = metrics_calculator.compute_genre_specific_metrics(
            predictions, labels, genres, probabilities
        )

        assert isinstance(metrics, dict)
        assert isinstance(genre_metrics, dict)