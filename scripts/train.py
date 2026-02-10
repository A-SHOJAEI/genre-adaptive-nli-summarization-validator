#!/usr/bin/env python3
"""Training script for genre-adaptive NLI summarization validator."""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from genre_adaptive_nli_summarization_validator.models.model import (
    GenreAdaptiveNLIValidator,
    GenreAdaptiveNLIConfig
)
from genre_adaptive_nli_summarization_validator.training.trainer import GenreAdaptiveTrainer
from genre_adaptive_nli_summarization_validator.data.loader import SummarizationDataLoader
from genre_adaptive_nli_summarization_validator.utils.config import Config


def setup_logging(config: Config) -> None:
    """Setup logging configuration.

    Args:
        config: Configuration object.
    """
    config.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting genre-adaptive NLI training")


def create_genre_mappings(datasets: dict) -> dict:
    """Create genre to ID mappings from datasets.

    Args:
        datasets: Dictionary of datasets.

    Returns:
        Dictionary mapping genre names to IDs.
    """
    all_genres = set()

    # Collect all unique genres
    for dataset in datasets.values():
        if dataset is not None:
            for example in dataset:
                all_genres.add(example.get("genre", "unknown"))

    # Create mapping
    genre_to_id = {genre: idx for idx, genre in enumerate(sorted(all_genres))}

    logging.info(f"Found {len(genre_to_id)} unique genres: {list(genre_to_id.keys())}")
    return genre_to_id


def load_and_prepare_data(config: Config) -> tuple:
    """Load and prepare training data.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset, genre_to_id).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading datasets...")

    # Initialize data loader
    data_loader = SummarizationDataLoader(
        tokenizer_name=config.get("model.name"),
        cache_dir=config.get("data.cache_dir"),
        seed=config.get("training.seed", 42)
    )

    # Load MultiNLI dataset for base NLI training
    logger.info("Loading MultiNLI dataset...")
    multinli_train = data_loader.load_multinli_dataset(
        split="train",
        max_samples=config.get("data.max_train_samples")
    )
    multinli_val = data_loader.load_multinli_dataset(
        split="validation_matched",
        max_samples=config.get("data.max_val_samples")
    )

    # Load CNN/DailyMail for summarization data
    logger.info("Loading CNN/DailyMail dataset...")
    cnn_train = data_loader.load_cnn_dailymail_dataset(
        split="train",
        max_samples=min(config.get("data.max_train_samples", 10000) or 10000, 5000)  # Limit for memory efficiency
    )
    cnn_val = data_loader.load_cnn_dailymail_dataset(
        split="validation",
        max_samples=min(config.get("data.max_val_samples", 2000) or 2000, 1000)
    )

    # Create NLI training data from summarization datasets
    logger.info("Creating NLI training data from summarization datasets...")
    summary_nli_train = data_loader.create_nli_training_data(
        cnn_train,
        negative_sampling_ratio=0.3,
        max_length=config.get("model.max_length", 512)
    )
    summary_nli_val = data_loader.create_nli_training_data(
        cnn_val,
        negative_sampling_ratio=0.3,
        max_length=config.get("model.max_length", 512)
    )

    # Combine datasets
    from datasets import concatenate_datasets, Features, Value, Sequence

    # Align features: MultiNLI uses ClassLabel for 'label', but summary NLI uses int64.
    # We need to cast MultiNLI 'label' to int64 and keep only shared columns.
    # First, identify common columns for concatenation
    shared_columns = set(multinli_train.column_names) & set(summary_nli_train.column_names)
    logger.info(f"MultiNLI columns: {multinli_train.column_names}")
    logger.info(f"Summary NLI columns: {summary_nli_train.column_names}")
    logger.info(f"Shared columns: {shared_columns}")

    # The summary NLI data already has tokenized columns (input_ids, attention_mask, etc.)
    # but MultiNLI does not. We need to tokenize MultiNLI too, or only use the summary NLI data.
    # Best approach: tokenize MultiNLI premises/hypotheses, then combine.
    # For now, use only the tokenized columns from summary NLI and prepare MultiNLI similarly.

    # Strategy: Use MultiNLI directly (it has premise/hypothesis/label/genre)
    # and tokenize it the same way as the summary NLI data.
    logger.info("Tokenizing MultiNLI data for compatibility...")

    def tokenize_multinli(example):
        """Tokenize MultiNLI example to match summary NLI format."""
        inputs = data_loader.preprocessor.prepare_model_inputs(
            example["premise"],
            example["hypothesis"],
            config.get("model.max_length", 512)
        )
        return {
            **inputs,
            "labels": int(example["label"]),
            "genre": example["genre"]
        }

    # Tokenize MultiNLI datasets
    multinli_train_tokenized = multinli_train.map(
        tokenize_multinli,
        desc="Tokenizing MultiNLI train",
        batched=False,
        load_from_cache_file=True
    )
    multinli_val_tokenized = multinli_val.map(
        tokenize_multinli,
        desc="Tokenizing MultiNLI val",
        batched=False,
        load_from_cache_file=True
    )

    # Keep only the columns needed for training
    keep_columns = ["input_ids", "attention_mask", "labels", "genre"]
    if "token_type_ids" in multinli_train_tokenized.column_names:
        keep_columns.append("token_type_ids")

    # Remove extra columns from both datasets
    multinli_train_cols_to_remove = [c for c in multinli_train_tokenized.column_names if c not in keep_columns]
    multinli_val_cols_to_remove = [c for c in multinli_val_tokenized.column_names if c not in keep_columns]
    summary_train_cols_to_remove = [c for c in summary_nli_train.column_names if c not in keep_columns]
    summary_val_cols_to_remove = [c for c in summary_nli_val.column_names if c not in keep_columns]

    multinli_train_tokenized = multinli_train_tokenized.remove_columns(multinli_train_cols_to_remove)
    multinli_val_tokenized = multinli_val_tokenized.remove_columns(multinli_val_cols_to_remove)
    summary_nli_train = summary_nli_train.remove_columns(summary_train_cols_to_remove)
    summary_nli_val = summary_nli_val.remove_columns(summary_val_cols_to_remove)

    # Cast labels to same type (int64) in both datasets
    from datasets import Features, Value, Sequence
    target_features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Value("int64"),
        "genre": Value("string"),
    })
    if "token_type_ids" in keep_columns:
        target_features["token_type_ids"] = Sequence(Value("int8"))

    multinli_train_tokenized = multinli_train_tokenized.cast(target_features)
    multinli_val_tokenized = multinli_val_tokenized.cast(target_features)
    summary_nli_train = summary_nli_train.cast(target_features)
    summary_nli_val = summary_nli_val.cast(target_features)

    # Combine training data
    combined_train = concatenate_datasets([multinli_train_tokenized, summary_nli_train])
    combined_val = concatenate_datasets([multinli_val_tokenized, summary_nli_val])

    # Create genre mappings
    all_datasets = {
        "train": combined_train,
        "val": combined_val
    }
    genre_to_id = create_genre_mappings(all_datasets)

    # Create balanced splits if specified
    if config.get("data.create_balanced_splits", True):
        logger.info("Creating genre-balanced splits...")
        splits = data_loader.create_genre_balanced_split(
            combined_train,
            test_size=0.1,
            val_size=0.1
        )
        train_dataset = splits["train"]
        eval_dataset = splits["validation"]
        test_dataset = splits["test"]
    else:
        train_dataset = combined_train
        eval_dataset = combined_val
        test_dataset = None

    # Log dataset statistics
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset) if eval_dataset else 0}")
    logger.info(f"Test samples: {len(test_dataset) if test_dataset else 0}")

    # Log genre distributions
    train_genre_stats = data_loader.get_genre_statistics(train_dataset)
    logger.info(f"Training genre distribution: {train_genre_stats}")

    return train_dataset, eval_dataset, test_dataset, genre_to_id


def create_model(config: Config, genre_to_id: dict) -> tuple:
    """Create model and tokenizer.

    Args:
        config: Configuration object.
        genre_to_id: Genre to ID mapping.

    Returns:
        Tuple of (model, tokenizer).
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating model...")

    # Create model configuration
    model_config = GenreAdaptiveNLIConfig(
        base_model_name=config.get("model.name", "microsoft/deberta-v3-base"),
        num_labels=config.get("model.num_labels", 3),
        num_genres=len(genre_to_id),
        genre_embedding_dim=config.get("model.genre_embedding_dim", 128),
        genre_adaptation_layers=config.get("model.genre_adaptation_layers", 2),
        dropout=config.get("model.dropout", 0.1),
        genre_attention_heads=config.get("model.genre_attention_heads", 8),
        cross_genre_regularization=config.get("model.cross_genre_regularization", 0.1),
        temperature_scaling=config.get("model.temperature_scaling", True)
    )

    # Create model
    model = GenreAdaptiveNLIValidator(model_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model.name", "microsoft/deberta-v3-base")
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train genre-adaptive NLI summarization validator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config.set("training.output_dir", args.output_dir)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Set random seeds for reproducibility
    seed = config.get("training.seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        # Load and prepare data
        train_dataset, eval_dataset, test_dataset, genre_to_id = load_and_prepare_data(config)

        # Create model and tokenizer
        model, tokenizer = create_model(config, genre_to_id)

        # Create trainer
        trainer = GenreAdaptiveTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            genre_to_id=genre_to_id
        )

        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            model.load_state_dict(torch.load(f"{args.resume_from}/pytorch_model.bin"))

        # Train model
        training_results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Evaluate on test set if available
        if test_dataset:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate_model(test_dataset, save_report=True)

            # Log test metrics
            target_metrics = test_results.get("overall_metrics", {})
            logger.info("Test Results:")
            logger.info(f"  Entailment AUC: {target_metrics.get('entailment_auc', 0.0):.4f}")
            logger.info(f"  Hallucination F1: {target_metrics.get('f1_entailment', 0.0):.4f}")
            logger.info(f"  Overall Accuracy: {target_metrics.get('accuracy', 0.0):.4f}")

            # Log genre-specific results
            genre_metrics = test_results.get("genre_metrics", {})
            logger.info("\nPer-Genre Results:")
            for genre, metrics in genre_metrics.items():
                logger.info(f"  {genre}: Acc={metrics.get('accuracy', 0.0):.4f}, "
                          f"F1={metrics.get('f1_macro', 0.0):.4f}")

        # Save genre mappings
        output_dir = Path(config.get("training.output_dir", "./checkpoints"))
        genre_mapping_path = output_dir / "genre_mappings.json"
        import json
        with open(genre_mapping_path, "w") as f:
            json.dump(genre_to_id, f, indent=2)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()