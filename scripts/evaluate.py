#!/usr/bin/env python3
"""Evaluation script for genre-adaptive NLI summarization validator."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from genre_adaptive_nli_summarization_validator.models.model import GenreAdaptiveNLIValidator
from genre_adaptive_nli_summarization_validator.data.loader import SummarizationDataLoader
from genre_adaptive_nli_summarization_validator.data.preprocessing import TextPreprocessor
from genre_adaptive_nli_summarization_validator.evaluation.metrics import SummaryValidationMetrics
from genre_adaptive_nli_summarization_validator.utils.config import Config


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_model_and_tokenizer(model_path: str) -> Tuple[GenreAdaptiveNLIValidator, AutoTokenizer, Dict[str, int]]:
    """Load trained model, tokenizer, and genre mappings.

    Args:
        model_path: Path to trained model directory.

    Returns:
        Tuple of (model, tokenizer, genre_to_id).
    """
    logger = logging.getLogger(__name__)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Load model
    model = GenreAdaptiveNLIValidator.from_pretrained(str(model_path))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Load genre mappings
    genre_mapping_path = model_path / "genre_mappings.json"
    if genre_mapping_path.exists():
        with open(genre_mapping_path, "r") as f:
            genre_to_id = json.load(f)
    else:
        logger.warning("Genre mappings not found, using default mappings")
        genre_to_id = {"unknown": 0}

    logger.info(f"Loaded model with {len(genre_to_id)} genres")
    return model, tokenizer, genre_to_id


def evaluate_on_dataset(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    dataset_name: str,
    split: str = "test",
    max_samples: int = None,
    output_dir: Path = None
) -> Dict:
    """Evaluate model on a specific dataset.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        dataset_name: Name of dataset to evaluate on.
        split: Dataset split to use.
        max_samples: Maximum samples to evaluate (optional).
        output_dir: Directory to save results (optional).

    Returns:
        Evaluation results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating on {dataset_name} {split} split")

    # Load dataset
    data_loader = SummarizationDataLoader(
        tokenizer_name=tokenizer.name_or_path,
        cache_dir="./data/cache"
    )

    if dataset_name.lower() == "multinli":
        if split == "test":
            split = "validation_mismatched"  # Use mismatched for test
        dataset = data_loader.load_multinli_dataset(split, max_samples)
    elif dataset_name.lower() == "cnn_dailymail":
        dataset = data_loader.load_cnn_dailymail_dataset(split, max_samples)
        # Create NLI data from summarization
        dataset = data_loader.create_nli_training_data(dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    labels = []
    probabilities = []
    genres = []

    logger.info(f"Processing {len(dataset)} samples...")

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(dataset)} samples")

            # Prepare inputs
            if "premise" in example and "hypothesis" in example:
                premise = example["premise"]
                hypothesis = example["hypothesis"]
            else:
                continue

            genre = example.get("genre", "unknown")
            label = example.get("labels", example.get("label", 0))

            # Get prediction
            result = model.predict_entailment_score(
                premise=premise,
                hypothesis=hypothesis,
                genre=genre,
                tokenizer=tokenizer,
                genre_to_id=genre_to_id,
                max_length=512
            )

            predictions.append(result["predicted_label"])
            labels.append(label)
            probabilities.append([
                result["entailment_score"],
                result["neutral_score"],
                result["contradiction_score"]
            ])
            genres.append(genre)

    # Calculate metrics
    metrics_calculator = SummaryValidationMetrics()
    evaluation_report = metrics_calculator.create_evaluation_report(
        predictions=predictions,
        labels=labels,
        genres=genres,
        probabilities=probabilities
    )

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed report
        report_path = output_dir / f"{dataset_name}_{split}_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(evaluation_report, f, indent=2)

        # Save predictions
        predictions_df = pd.DataFrame({
            "prediction": predictions,
            "label": labels,
            "genre": genres,
            "entailment_prob": [p[0] for p in probabilities],
            "neutral_prob": [p[1] for p in probabilities],
            "contradiction_prob": [p[2] for p in probabilities]
        })
        predictions_path = output_dir / f"{dataset_name}_{split}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

        # Create and save plots
        metrics_calculator.plot_confusion_matrix(
            predictions, labels,
            save_path=output_dir / f"{dataset_name}_{split}_confusion_matrix.png"
        )

        metrics_calculator.plot_genre_performance(
            evaluation_report["genre_metrics"],
            metric_name="accuracy",
            save_path=output_dir / f"{dataset_name}_{split}_genre_accuracy.png"
        )

        logger.info(f"Results saved to {output_dir}")

    return evaluation_report


def evaluate_summary_validation(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    documents: List[str],
    summaries: List[str],
    genres: List[str],
    output_dir: Path = None
) -> Dict:
    """Evaluate model on summary validation task.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        documents: List of source documents.
        summaries: List of summaries to validate.
        genres: List of document genres.
        output_dir: Directory to save results (optional).

    Returns:
        Validation results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating summary validation on {len(documents)} documents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    validation_results = []

    for i, (document, summary, genre) in enumerate(zip(documents, summaries, genres)):
        if i % 50 == 0:
            logger.info(f"Validated {i}/{len(documents)} summaries")

        # Validate summary
        result = model.validate_summary(
            document=document,
            summary=summary,
            genre=genre,
            tokenizer=tokenizer,
            genre_to_id=genre_to_id,
            threshold=0.5
        )

        validation_results.append({
            "document_id": i,
            "is_valid": result["is_valid"],
            "entailment_score": result["entailment_score"],
            "confidence": result["confidence"],
            "entropy": result["entropy"],
            "genre": genre,
            "summary_length": result["summary_length"],
            "document_length": result["document_length"],
            "compression_ratio": result["compression_ratio"]
        })

    # Aggregate results
    valid_count = sum(1 for r in validation_results if r["is_valid"])
    total_count = len(validation_results)

    # Calculate metrics by genre
    genre_stats = {}
    for genre in set(genres):
        genre_results = [r for r in validation_results if r["genre"] == genre]
        genre_valid = sum(1 for r in genre_results if r["is_valid"])
        genre_total = len(genre_results)

        genre_stats[genre] = {
            "validation_rate": genre_valid / genre_total if genre_total > 0 else 0,
            "avg_entailment_score": sum(r["entailment_score"] for r in genre_results) / genre_total if genre_total > 0 else 0,
            "avg_confidence": sum(r["confidence"] for r in genre_results) / genre_total if genre_total > 0 else 0,
            "sample_count": genre_total
        }

    summary_validation_report = {
        "overall_validation_rate": valid_count / total_count,
        "total_summaries": total_count,
        "valid_summaries": valid_count,
        "invalid_summaries": total_count - valid_count,
        "genre_statistics": genre_stats,
        "detailed_results": validation_results
    }

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary validation report
        report_path = output_dir / "summary_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(summary_validation_report, f, indent=2)

        # Save detailed results
        results_df = pd.DataFrame(validation_results)
        results_path = output_dir / "summary_validation_results.csv"
        results_df.to_csv(results_path, index=False)

        logger.info(f"Summary validation results saved to {output_dir}")

    return summary_validation_report


def benchmark_performance(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    output_dir: Path = None
) -> Dict:
    """Run comprehensive benchmark evaluation.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        output_dir: Directory to save results (optional).

    Returns:
        Benchmark results dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running comprehensive benchmark evaluation")

    benchmark_results = {}

    # Evaluate on MultiNLI
    try:
        multinli_results = evaluate_on_dataset(
            model, tokenizer, genre_to_id,
            dataset_name="multinli",
            split="test",
            max_samples=5000,
            output_dir=output_dir / "multinli" if output_dir else None
        )
        benchmark_results["multinli"] = multinli_results["overall_metrics"]
    except Exception as e:
        logger.error(f"MultiNLI evaluation failed: {e}")
        benchmark_results["multinli"] = {"error": str(e)}

    # Evaluate on CNN/DailyMail
    try:
        cnn_results = evaluate_on_dataset(
            model, tokenizer, genre_to_id,
            dataset_name="cnn_dailymail",
            split="test",
            max_samples=2000,
            output_dir=output_dir / "cnn_dailymail" if output_dir else None
        )
        benchmark_results["cnn_dailymail"] = cnn_results["overall_metrics"]
    except Exception as e:
        logger.error(f"CNN/DailyMail evaluation failed: {e}")
        benchmark_results["cnn_dailymail"] = {"error": str(e)}

    # Calculate target metrics
    target_metrics = {}

    # Entailment AUC (from MultiNLI)
    if "multinli" in benchmark_results and "entailment_auc" in benchmark_results["multinli"]:
        target_metrics["entailment_auc"] = benchmark_results["multinli"]["entailment_auc"]

    # Hallucination detection F1 (from CNN/DailyMail)
    if "cnn_dailymail" in benchmark_results and "f1_entailment" in benchmark_results["cnn_dailymail"]:
        target_metrics["hallucination_detection_f1"] = benchmark_results["cnn_dailymail"]["f1_entailment"]

    # Genre transfer accuracy (minimum across genres)
    genre_accuracies = []
    for dataset_results in benchmark_results.values():
        if isinstance(dataset_results, dict) and "genre_metrics" in multinli_results:
            for genre_metrics in multinli_results["genre_metrics"].values():
                if "accuracy" in genre_metrics:
                    genre_accuracies.append(genre_metrics["accuracy"])

    if genre_accuracies:
        target_metrics["genre_transfer_accuracy"] = min(genre_accuracies)

    benchmark_results["target_metrics"] = target_metrics

    # Save benchmark results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = output_dir / "benchmark_results.json"
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        # Print target metrics comparison
        target_goals = {
            "entailment_auc": 0.85,
            "hallucination_detection_f1": 0.78,
            "genre_transfer_accuracy": 0.72
        }

        logger.info("\n" + "="*50)
        logger.info("TARGET METRICS COMPARISON")
        logger.info("="*50)
        for metric, goal in target_goals.items():
            achieved = target_metrics.get(metric, 0.0)
            status = "✓ PASS" if achieved >= goal else "✗ FAIL"
            logger.info(f"{metric:25}: {achieved:.4f} / {goal:.4f} {status}")
        logger.info("="*50)

    return benchmark_results


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate genre-adaptive NLI summarization validator")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["multinli", "cnn_dailymail", "all", "benchmark"],
        default="benchmark",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--custom-data",
        type=str,
        help="Path to custom CSV file with documents and summaries"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load model, tokenizer, and genre mappings
        model, tokenizer, genre_to_id = load_model_and_tokenizer(args.model_path)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.dataset == "benchmark":
            # Run comprehensive benchmark
            benchmark_results = benchmark_performance(
                model, tokenizer, genre_to_id, output_dir
            )

        elif args.dataset == "all":
            # Evaluate on all datasets
            for dataset_name in ["multinli", "cnn_dailymail"]:
                try:
                    evaluate_on_dataset(
                        model, tokenizer, genre_to_id,
                        dataset_name, args.split, args.max_samples,
                        output_dir / dataset_name
                    )
                except Exception as e:
                    logger.error(f"Failed to evaluate on {dataset_name}: {e}")

        elif args.custom_data:
            # Evaluate on custom data
            logger.info(f"Loading custom data from {args.custom_data}")
            df = pd.read_csv(args.custom_data)

            required_cols = ["document", "summary"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Custom data must have columns: {required_cols}")

            documents = df["document"].tolist()
            summaries = df["summary"].tolist()
            genres = df.get("genre", ["unknown"] * len(documents)).tolist()

            validation_results = evaluate_summary_validation(
                model, tokenizer, genre_to_id,
                documents, summaries, genres, output_dir
            )

        else:
            # Evaluate on specific dataset
            evaluation_results = evaluate_on_dataset(
                model, tokenizer, genre_to_id,
                args.dataset, args.split, args.max_samples, output_dir
            )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()