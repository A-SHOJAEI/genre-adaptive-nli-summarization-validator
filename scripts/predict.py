#!/usr/bin/env python3
"""Inference script for genre-adaptive NLI summarization validator."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from genre_adaptive_nli_summarization_validator.models.model import GenreAdaptiveNLIValidator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_model_and_tokenizer(model_path: str) -> tuple:
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
        logger.warning("Genre mappings not found, using default")
        genre_to_id = {"unknown": 0}

    return model, tokenizer, genre_to_id


def predict_single(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    premise: str,
    hypothesis: str,
    genre: str = "unknown"
) -> Dict:
    """Run prediction on a single example.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        premise: Source document or premise text.
        hypothesis: Summary or hypothesis text.
        genre: Text genre.

    Returns:
        Dictionary containing prediction results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        result = model.predict_entailment_score(
            premise=premise,
            hypothesis=hypothesis,
            genre=genre,
            tokenizer=tokenizer,
            genre_to_id=genre_to_id,
            max_length=512
        )

    return result


def predict_summary_validation(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    document: str,
    summary: str,
    genre: str = "unknown",
    threshold: float = 0.5
) -> Dict:
    """Validate a summary against a document.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        document: Source document.
        summary: Generated summary to validate.
        genre: Document genre.
        threshold: Entailment threshold for validation.

    Returns:
        Dictionary containing validation results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        result = model.validate_summary(
            document=document,
            summary=summary,
            genre=genre,
            tokenizer=tokenizer,
            genre_to_id=genre_to_id,
            threshold=threshold
        )

    return result


def predict_batch(
    model: GenreAdaptiveNLIValidator,
    tokenizer: AutoTokenizer,
    genre_to_id: Dict[str, int],
    examples: List[Dict],
    mode: str = "nli"
) -> List[Dict]:
    """Run predictions on a batch of examples.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        genre_to_id: Genre to ID mapping.
        examples: List of example dictionaries.
        mode: Prediction mode ("nli" or "summary").

    Returns:
        List of prediction results.
    """
    logger = logging.getLogger(__name__)
    results = []

    for i, example in enumerate(examples):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{len(examples)}")

        try:
            if mode == "nli":
                result = predict_single(
                    model, tokenizer, genre_to_id,
                    premise=example.get("premise", ""),
                    hypothesis=example.get("hypothesis", ""),
                    genre=example.get("genre", "unknown")
                )
            else:  # summary mode
                result = predict_summary_validation(
                    model, tokenizer, genre_to_id,
                    document=example.get("document", ""),
                    summary=example.get("summary", ""),
                    genre=example.get("genre", "unknown"),
                    threshold=example.get("threshold", 0.5)
                )

            results.append({
                "example_id": i,
                **result,
                "input": example
            })
        except Exception as e:
            logger.error(f"Failed to process example {i}: {e}")
            results.append({
                "example_id": i,
                "error": str(e),
                "input": example
            })

    return results


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Run inference with genre-adaptive NLI summarization validator"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/final-model",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["nli", "summary", "interactive"],
        default="interactive",
        help="Prediction mode"
    )
    parser.add_argument(
        "--premise",
        type=str,
        help="Premise text (for NLI mode)"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        help="Hypothesis text (for NLI mode)"
    )
    parser.add_argument(
        "--document",
        type=str,
        help="Source document (for summary mode)"
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Summary text (for summary mode)"
    )
    parser.add_argument(
        "--genre",
        type=str,
        default="unknown",
        help="Text genre"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entailment threshold for summary validation"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="JSON file with batch of examples"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for predictions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load model
        model, tokenizer, genre_to_id = load_model_and_tokenizer(args.model_path)
        logger.info(f"Available genres: {list(genre_to_id.keys())}")

        # Run predictions based on mode
        if args.input_file:
            # Batch mode
            logger.info(f"Loading examples from {args.input_file}")
            with open(args.input_file, "r") as f:
                examples = json.load(f)

            results = predict_batch(
                model, tokenizer, genre_to_id,
                examples, mode=args.mode
            )

            # Save results
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output_file}")
            else:
                print(json.dumps(results, indent=2))

        elif args.mode == "interactive":
            # Interactive mode
            print("\n" + "="*70)
            print("Genre-Adaptive NLI Summarization Validator - Interactive Mode")
            print("="*70)
            print(f"Available genres: {', '.join(genre_to_id.keys())}")
            print("\nChoose validation type:")
            print("  1. NLI (premise + hypothesis)")
            print("  2. Summary validation (document + summary)")

            choice = input("\nEnter choice (1 or 2): ").strip()

            if choice == "1":
                premise = input("\nEnter premise: ").strip()
                hypothesis = input("Enter hypothesis: ").strip()
                genre = input(f"Enter genre [{args.genre}]: ").strip() or args.genre

                result = predict_single(
                    model, tokenizer, genre_to_id,
                    premise, hypothesis, genre
                )

                print("\n" + "="*70)
                print("PREDICTION RESULTS")
                print("="*70)
                print(f"Predicted Label: {result['predicted_label']}")
                print(f"Entailment Score: {result['entailment_score']:.4f}")
                print(f"Neutral Score: {result['neutral_score']:.4f}")
                print(f"Contradiction Score: {result['contradiction_score']:.4f}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("="*70)

            elif choice == "2":
                document = input("\nEnter source document: ").strip()
                summary = input("Enter summary: ").strip()
                genre = input(f"Enter genre [{args.genre}]: ").strip() or args.genre
                threshold = input(f"Enter threshold [{args.threshold}]: ").strip()
                threshold = float(threshold) if threshold else args.threshold

                result = predict_summary_validation(
                    model, tokenizer, genre_to_id,
                    document, summary, genre, threshold
                )

                print("\n" + "="*70)
                print("SUMMARY VALIDATION RESULTS")
                print("="*70)
                print(f"Valid Summary: {'YES' if result['is_valid'] else 'NO'}")
                print(f"Entailment Score: {result['entailment_score']:.4f}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Entropy: {result['entropy']:.4f}")
                print(f"Compression Ratio: {result['compression_ratio']:.2f}")
                print("="*70)

        elif args.mode == "nli":
            # Single NLI prediction
            if not args.premise or not args.hypothesis:
                raise ValueError("--premise and --hypothesis required for NLI mode")

            result = predict_single(
                model, tokenizer, genre_to_id,
                args.premise, args.hypothesis, args.genre
            )

            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))

        elif args.mode == "summary":
            # Single summary validation
            if not args.document or not args.summary:
                raise ValueError("--document and --summary required for summary mode")

            result = predict_summary_validation(
                model, tokenizer, genre_to_id,
                args.document, args.summary, args.genre, args.threshold
            )

            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                print(json.dumps(result, indent=2))

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
