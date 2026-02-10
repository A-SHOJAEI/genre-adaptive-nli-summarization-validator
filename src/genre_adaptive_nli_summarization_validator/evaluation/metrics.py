"""Evaluation metrics for genre-adaptive NLI summarization validation."""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class SummaryValidationMetrics:
    """Comprehensive evaluation metrics for summary validation using NLI."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
        self.label_names = ["entailment", "neutral", "contradiction"]

    def compute_basic_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """Compute basic classification metrics.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            probabilities: Prediction probabilities (optional).

        Returns:
            Dictionary of basic metrics.
        """
        metrics = {}

        # Accuracy
        metrics["accuracy"] = accuracy_score(labels, predictions)

        # Precision, recall, F1 (macro and micro)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["precision_macro"] = precision
        metrics["recall_macro"] = recall
        metrics["f1_macro"] = f1

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["precision_micro"] = precision_micro
        metrics["recall_micro"] = recall_micro
        metrics["f1_micro"] = f1_micro

        # Class-specific metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        for i, label_name in enumerate(self.label_names):
            if i < len(precision_per_class):
                metrics[f"precision_{label_name}"] = precision_per_class[i]
                metrics[f"recall_{label_name}"] = recall_per_class[i]
                metrics[f"f1_{label_name}"] = f1_per_class[i]

        # AUC metrics (if probabilities provided)
        if probabilities is not None:
            probabilities = np.array(probabilities)

            # Binary classification for entailment detection
            entailment_labels = [1 if label == 0 else 0 for label in labels]
            entailment_probs = probabilities[:, 0]
            metrics["entailment_auc"] = roc_auc_score(entailment_labels, entailment_probs)
            metrics["entailment_ap"] = average_precision_score(entailment_labels, entailment_probs)

            # Multi-class AUC (one-vs-rest)
            try:
                metrics["auc_ovr"] = roc_auc_score(labels, probabilities, multi_class="ovr")
            except ValueError:
                metrics["auc_ovr"] = 0.0

        return metrics

    def compute_genre_specific_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        genres: List[str],
        probabilities: Optional[List[List[float]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per genre.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            genres: Genre labels.
            probabilities: Prediction probabilities (optional).

        Returns:
            Dictionary of metrics per genre.
        """
        genre_metrics = {}
        unique_genres = list(set(genres))

        for genre in unique_genres:
            # Filter predictions/labels for this genre
            genre_mask = [i for i, g in enumerate(genres) if g == genre]

            if not genre_mask:
                continue

            genre_predictions = [predictions[i] for i in genre_mask]
            genre_labels = [labels[i] for i in genre_mask]
            genre_probs = [probabilities[i] for i in genre_mask] if probabilities else None

            # Compute metrics for this genre
            genre_metrics[genre] = self.compute_basic_metrics(
                genre_predictions, genre_labels, genre_probs
            )

            # Add sample count
            genre_metrics[genre]["sample_count"] = len(genre_mask)

        return genre_metrics

    def compute_hallucination_detection_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: List[List[float]],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute metrics specifically for hallucination detection.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            probabilities: Prediction probabilities.
            threshold: Entailment threshold for binary classification.

        Returns:
            Hallucination detection metrics.
        """
        probabilities = np.array(probabilities)

        # Binary classification: entailment (valid) vs. non-entailment (hallucination)
        valid_labels = [1 if label == 0 else 0 for label in labels]  # 1 for entailment
        entailment_probs = probabilities[:, 0]
        valid_predictions = [1 if prob >= threshold else 0 for prob in entailment_probs]

        metrics = {}

        # Basic binary metrics
        metrics["hallucination_detection_accuracy"] = accuracy_score(valid_labels, valid_predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels, valid_predictions, average="binary", pos_label=0, zero_division=0
        )
        metrics["hallucination_detection_precision"] = precision
        metrics["hallucination_detection_recall"] = recall
        metrics["hallucination_detection_f1"] = f1

        # AUC for hallucination detection (treat non-entailment as positive)
        hallucination_labels = [0 if label == 0 else 1 for label in labels]
        hallucination_scores = 1 - entailment_probs  # Higher score = more likely hallucination

        if len(set(hallucination_labels)) > 1:
            metrics["hallucination_auc"] = roc_auc_score(hallucination_labels, hallucination_scores)
            metrics["hallucination_ap"] = average_precision_score(hallucination_labels, hallucination_scores)

        return metrics

    def compute_calibration_metrics(
        self,
        predictions: List[int],
        probabilities: List[List[float]],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration metrics for confidence estimation.

        Args:
            predictions: Predicted labels.
            probabilities: Prediction probabilities.
            n_bins: Number of bins for calibration curve.

        Returns:
            Calibration metrics and data.
        """
        probabilities = np.array(probabilities)
        max_probs = np.max(probabilities, axis=1)
        confidences = max_probs
        accuracies = (predictions == np.argmax(probabilities, axis=1)).astype(int)

        # Reliability diagram data
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=n_bins, strategy="uniform"
        )

        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        # Maximum Calibration Error (MCE)
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "reliability_diagram_data": {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist()
            }
        }

    def compute_genre_transfer_metrics(
        self,
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """Compute genre transfer performance metrics.

        Args:
            source_metrics: Overall model performance metrics.
            target_metrics: Per-genre performance metrics.
            baseline_metrics: Baseline per-genre metrics (optional).

        Returns:
            Genre transfer metrics.
        """
        transfer_metrics = {}

        # Genre-specific accuracy variance
        genre_accuracies = [metrics.get("accuracy", 0.0) for metrics in target_metrics.values()]
        transfer_metrics["genre_accuracy_mean"] = np.mean(genre_accuracies)
        transfer_metrics["genre_accuracy_std"] = np.std(genre_accuracies)

        # Relative standard deviation (coefficient of variation)
        if transfer_metrics["genre_accuracy_mean"] > 0:
            transfer_metrics["genre_accuracy_cv"] = (
                transfer_metrics["genre_accuracy_std"] / transfer_metrics["genre_accuracy_mean"]
            )

        # Genre adaptation effectiveness (compared to baseline)
        if baseline_metrics:
            improvements = []
            for genre in target_metrics:
                if genre in baseline_metrics:
                    current_acc = target_metrics[genre].get("accuracy", 0.0)
                    baseline_acc = baseline_metrics[genre].get("accuracy", 0.0)
                    if baseline_acc > 0:
                        improvement = (current_acc - baseline_acc) / baseline_acc
                        improvements.append(improvement)

            if improvements:
                transfer_metrics["mean_improvement"] = np.mean(improvements)
                transfer_metrics["improvement_consistency"] = 1 - np.std(improvements)

        # Cross-genre consistency
        f1_scores = [metrics.get("f1_macro", 0.0) for metrics in target_metrics.values()]
        transfer_metrics["genre_f1_consistency"] = 1 - np.std(f1_scores) / max(np.mean(f1_scores), 1e-8)

        return transfer_metrics

    def create_evaluation_report(
        self,
        predictions: List[int],
        labels: List[int],
        genres: List[str],
        probabilities: Optional[List[List[float]]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive evaluation report.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            genres: Genre labels.
            probabilities: Prediction probabilities (optional).
            save_path: Path to save detailed report (optional).

        Returns:
            Complete evaluation report.
        """
        report = {}

        # Basic metrics
        report["overall_metrics"] = self.compute_basic_metrics(predictions, labels, probabilities)

        # Genre-specific metrics
        report["genre_metrics"] = self.compute_genre_specific_metrics(
            predictions, labels, genres, probabilities
        )

        # Hallucination detection metrics
        if probabilities:
            report["hallucination_metrics"] = self.compute_hallucination_detection_metrics(
                predictions, labels, probabilities
            )

            # Calibration metrics
            report["calibration_metrics"] = self.compute_calibration_metrics(predictions, probabilities)

        # Genre transfer metrics
        report["transfer_metrics"] = self.compute_genre_transfer_metrics(
            report["overall_metrics"],
            report["genre_metrics"]
        )

        # Confusion matrix
        report["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()

        # Classification report
        report["classification_report"] = classification_report(
            labels, predictions, target_names=self.label_names, output_dict=True
        )

        # Genre distribution
        genre_dist = {}
        for genre in genres:
            genre_dist[genre] = genre_dist.get(genre, 0) + 1
        report["genre_distribution"] = genre_dist

        # Sample statistics
        report["sample_statistics"] = {
            "total_samples": len(predictions),
            "num_genres": len(set(genres)),
            "label_distribution": {
                label: labels.count(i) for i, label in enumerate(self.label_names)
            }
        }

        # Save detailed report if path provided
        if save_path:
            self._save_detailed_report(report, save_path)

        return report

    def _save_detailed_report(self, report: Dict[str, Any], save_path: str) -> None:
        """Save detailed evaluation report to file.

        Args:
            report: Evaluation report dictionary.
            save_path: Path to save report.
        """
        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Detailed evaluation report saved to {save_path}")

    def plot_confusion_matrix(
        self,
        predictions: List[int],
        labels: List[int],
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> None:
        """Plot confusion matrix.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            save_path: Path to save plot (optional).
            normalize: Whether to normalize values.
        """
        cm = confusion_matrix(labels, predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_genre_performance(
        self,
        genre_metrics: Dict[str, Dict[str, float]],
        metric_name: str = "accuracy",
        save_path: Optional[str] = None
    ) -> None:
        """Plot performance across genres.

        Args:
            genre_metrics: Per-genre metrics.
            metric_name: Metric to plot.
            save_path: Path to save plot (optional).
        """
        genres = list(genre_metrics.keys())
        values = [genre_metrics[genre].get(metric_name, 0.0) for genre in genres]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(genres, values, alpha=0.7)
        plt.title(f'{metric_name.title()} by Genre')
        plt.xlabel('Genre')
        plt.ylabel(metric_name.title())
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def compute_target_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        genres: List[str],
        probabilities: Optional[List[List[float]]] = None
    ) -> Dict[str, float]:
        """Compute target metrics specified in the project requirements.

        Args:
            predictions: Predicted labels.
            labels: Ground truth labels.
            genres: Genre labels.
            probabilities: Prediction probabilities.

        Returns:
            Target metrics dictionary.
        """
        target_metrics = {}

        # Entailment AUC
        if probabilities is not None:
            entailment_labels = [1 if label == 0 else 0 for label in labels]
            entailment_probs = np.array(probabilities)[:, 0]
            target_metrics["entailment_auc"] = roc_auc_score(entailment_labels, entailment_probs)

        # Hallucination detection F1
        hallucination_metrics = self.compute_hallucination_detection_metrics(
            predictions, labels, probabilities
        )
        target_metrics["hallucination_detection_f1"] = hallucination_metrics["hallucination_detection_f1"]

        # Genre transfer accuracy (minimum accuracy across genres)
        genre_metrics = self.compute_genre_specific_metrics(predictions, labels, genres, probabilities)
        genre_accuracies = [metrics["accuracy"] for metrics in genre_metrics.values()]
        target_metrics["genre_transfer_accuracy"] = min(genre_accuracies) if genre_accuracies else 0.0

        return target_metrics