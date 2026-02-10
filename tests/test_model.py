"""Tests for genre-adaptive NLI model implementation."""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from genre_adaptive_nli_summarization_validator.models.model import (
    GenreAdaptiveNLIValidator,
    GenreAdaptiveNLIConfig,
    GenreAdaptationLayer
)
from conftest import assert_tensor_shape, assert_tensor_dtype, create_mock_batch


class TestGenreAdaptiveNLIConfig:
    """Test cases for GenreAdaptiveNLIConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenreAdaptiveNLIConfig()

        assert config.base_model_name == "microsoft/deberta-v3-base"
        assert config.num_labels == 3
        assert config.num_genres == 10
        assert config.genre_embedding_dim == 128
        assert config.genre_adaptation_layers == 2
        assert config.dropout == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenreAdaptiveNLIConfig(
            base_model_name="bert-base-uncased",
            num_labels=5,
            num_genres=8,
            genre_embedding_dim=256,
            genre_adaptation_layers=3,
            dropout=0.2
        )

        assert config.base_model_name == "bert-base-uncased"
        assert config.num_labels == 5
        assert config.num_genres == 8
        assert config.genre_embedding_dim == 256
        assert config.genre_adaptation_layers == 3
        assert config.dropout == 0.2

    def test_config_inheritance(self):
        """Test that config inherits from PretrainedConfig."""
        config = GenreAdaptiveNLIConfig()
        assert hasattr(config, 'model_type')
        assert config.model_type == "genre_adaptive_nli"


class TestGenreAdaptationLayer:
    """Test cases for GenreAdaptationLayer class."""

    @pytest.fixture
    def adaptation_layer(self):
        """Create sample adaptation layer."""
        return GenreAdaptationLayer(
            hidden_size=128,
            genre_embedding_dim=64,
            num_attention_heads=4,
            dropout=0.1
        )

    def test_init(self, adaptation_layer):
        """Test adaptation layer initialization."""
        assert adaptation_layer.hidden_size == 128
        assert adaptation_layer.genre_embedding_dim == 64
        assert adaptation_layer.num_attention_heads == 4

        # Check submodules exist
        assert isinstance(adaptation_layer.genre_attention, nn.MultiheadAttention)
        assert isinstance(adaptation_layer.genre_projection, nn.Linear)
        assert isinstance(adaptation_layer.adaptation_gate, nn.Linear)
        assert isinstance(adaptation_layer.ffn, nn.Sequential)

    def test_forward_pass(self, adaptation_layer, device):
        """Test forward pass through adaptation layer."""
        batch_size, seq_len, hidden_size = 2, 32, 128
        genre_embedding_dim = 64

        # Create input tensors
        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device)
        genre_embeddings = torch.randn(batch_size, genre_embedding_dim).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)

        adaptation_layer = adaptation_layer.to(device)

        # Forward pass
        output = adaptation_layer(hidden_states, genre_embeddings, attention_mask)

        # Check output shape and type
        assert_tensor_shape(output, (batch_size, seq_len, hidden_size), "adaptation_output")
        assert_tensor_dtype(output, torch.float32, "adaptation_output")

    def test_forward_without_attention_mask(self, adaptation_layer, device):
        """Test forward pass without attention mask."""
        batch_size, seq_len, hidden_size = 2, 32, 128
        genre_embedding_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device)
        genre_embeddings = torch.randn(batch_size, genre_embedding_dim).to(device)

        adaptation_layer = adaptation_layer.to(device)

        # Should work without attention mask
        output = adaptation_layer(hidden_states, genre_embeddings)
        assert_tensor_shape(output, (batch_size, seq_len, hidden_size), "adaptation_output")

    def test_gradient_flow(self, adaptation_layer, device):
        """Test that gradients flow through adaptation layer."""
        batch_size, seq_len, hidden_size = 2, 32, 128
        genre_embedding_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True).to(device)
        genre_embeddings = torch.randn(batch_size, genre_embedding_dim, requires_grad=True).to(device)

        adaptation_layer = adaptation_layer.to(device)

        output = adaptation_layer(hidden_states, genre_embeddings)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert genre_embeddings.grad is not None

        # Check parameter gradients
        for param in adaptation_layer.parameters():
            assert param.grad is not None


class TestGenreAdaptiveNLIValidator:
    """Test cases for GenreAdaptiveNLIValidator class."""

    def test_model_init(self, sample_model_config, sample_model):
        """Test model initialization."""
        model = sample_model

        # Check model components
        assert hasattr(model, 'base_model')
        assert hasattr(model, 'genre_embeddings')
        assert hasattr(model, 'adaptation_layers')
        assert hasattr(model, 'classifier')

        # Check parameter counts
        assert len(list(model.adaptation_layers)) == sample_model_config.genre_adaptation_layers

    def test_forward_pass_complete(self, sample_model, genre_to_id_mapping, device):
        """Test complete forward pass through model."""
        model = sample_model.to(device)
        batch = create_mock_batch(batch_size=2, seq_len=32, num_labels=3)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch, return_dict=True)

        # Check outputs
        assert "loss" in outputs
        assert "logits" in outputs
        assert "hidden_states" in outputs
        assert "pooled_output" in outputs

        # Check shapes
        batch_size, num_labels = 2, 3
        assert_tensor_shape(outputs["logits"], (batch_size, num_labels), "logits")
        assert_tensor_shape(outputs["pooled_output"], (batch_size, model.hidden_size), "pooled_output")

        # Check loss is scalar
        assert outputs["loss"].dim() == 0

    def test_forward_without_genre_ids(self, sample_model, device):
        """Test forward pass without genre IDs."""
        model = sample_model.to(device)
        batch = create_mock_batch(batch_size=2, seq_len=32)
        del batch["genre_ids"]  # Remove genre IDs
        batch = {k: v.to(device) for k, v in batch.items()}

        # Should still work without genre adaptation
        outputs = model(**batch, return_dict=True)
        assert "logits" in outputs

    def test_forward_without_labels(self, sample_model, device):
        """Test forward pass without labels (inference mode)."""
        model = sample_model.to(device)
        batch = create_mock_batch(batch_size=2, seq_len=32)
        del batch["labels"]  # Remove labels
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, return_dict=True)
        assert outputs["loss"] is None
        assert "logits" in outputs

    def test_predict_entailment_score(self, sample_model, sample_tokenizer, genre_to_id_mapping, device):
        """Test entailment score prediction."""
        model = sample_model.to(device)

        premise = "The cat is sleeping on the couch."
        hypothesis = "A cat is resting on furniture."
        genre = "fiction"

        result = model.predict_entailment_score(
            premise=premise,
            hypothesis=hypothesis,
            genre=genre,
            tokenizer=sample_tokenizer,
            genre_to_id=genre_to_id_mapping,
            device=device
        )

        # Check result structure
        assert isinstance(result, dict)
        required_keys = [
            "entailment_score", "neutral_score", "contradiction_score",
            "confidence", "entropy", "predicted_label", "genre"
        ]
        for key in required_keys:
            assert key in result

        # Check value ranges
        assert 0 <= result["entailment_score"] <= 1
        assert 0 <= result["neutral_score"] <= 1
        assert 0 <= result["contradiction_score"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["entropy"] >= 0
        assert result["predicted_label"] in [0, 1, 2]

    def test_validate_summary(self, sample_model, sample_tokenizer, genre_to_id_mapping, device):
        """Test summary validation functionality."""
        model = sample_model.to(device)

        document = "Scientists discovered a new species of bird in the Amazon rainforest."
        summary = "A new bird species was found in the Amazon."
        genre = "news"

        result = model.validate_summary(
            document=document,
            summary=summary,
            genre=genre,
            tokenizer=sample_tokenizer,
            genre_to_id=genre_to_id_mapping
        )

        # Check result structure
        assert isinstance(result, dict)
        required_keys = [
            "is_valid", "threshold_used", "summary_length", "document_length",
            "compression_ratio", "entailment_score"
        ]
        for key in required_keys:
            assert key in result

        # Check value types and ranges
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["threshold_used"], float)
        assert isinstance(result["summary_length"], int)
        assert isinstance(result["document_length"], int)
        assert 0 <= result["compression_ratio"] <= 1

    def test_temperature_scaling(self, genre_to_id_mapping):
        """Test temperature scaling functionality."""
        # Test with temperature scaling enabled
        config_with_temp = GenreAdaptiveNLIConfig(
            num_genres=len(genre_to_id_mapping),
            temperature_scaling=True
        )
        model_with_temp = GenreAdaptiveNLIValidator(config_with_temp)
        assert model_with_temp.temperature is not None

        # Test with temperature scaling disabled
        config_without_temp = GenreAdaptiveNLIConfig(
            num_genres=len(genre_to_id_mapping),
            temperature_scaling=False
        )
        model_without_temp = GenreAdaptiveNLIValidator(config_without_temp)
        assert model_without_temp.temperature is None

    def test_cross_genre_regularization(self, sample_model, device):
        """Test cross-genre regularization loss computation."""
        model = sample_model.to(device)

        # Create batch with different genres and labels
        batch_size = 4
        genre_ids = torch.tensor([0, 1, 0, 1]).to(device)  # Two different genres
        labels = torch.tensor([0, 0, 1, 1]).to(device)    # Same labels within genres
        logits = torch.randn(batch_size, 3).to(device)

        # Compute regularization loss
        reg_loss = model._compute_cross_genre_regularization(genre_ids, labels, logits)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.dim() == 0  # Should be scalar
        assert reg_loss.item() >= 0  # Should be non-negative

    def test_model_save_and_load(self, sample_model, temp_dir):
        """Test model saving and loading."""
        model = sample_model

        # Save model
        save_path = temp_dir / "test_model"
        model.save_pretrained(save_path)

        # Check files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "pytorch_model.bin").exists()

        # Load model
        loaded_model = GenreAdaptiveNLIValidator.from_pretrained(str(save_path))

        # Check model is loaded correctly
        assert loaded_model.config.num_labels == model.config.num_labels
        assert loaded_model.config.num_genres == model.config.num_genres

    def test_gradient_flow_complete_model(self, sample_model, device):
        """Test gradient flow through complete model."""
        model = sample_model.to(device)
        batch = create_mock_batch(batch_size=2, seq_len=32)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch, return_dict=True)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Check gradients exist for key components
        assert model.genre_embeddings.weight.grad is not None
        assert model.classifier.weight.grad is not None

        # Check adaptation layer gradients
        for layer in model.adaptation_layers:
            for param in layer.parameters():
                assert param.grad is not None

    def test_model_eval_mode(self, sample_model, device):
        """Test model behavior in eval mode."""
        model = sample_model.to(device)
        batch = create_mock_batch(batch_size=2, seq_len=32)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Test in training mode
        model.train()
        outputs_train = model(**batch, return_dict=True)

        # Test in eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(**batch, return_dict=True)

        # Outputs should have same shape but potentially different values due to dropout
        assert outputs_train["logits"].shape == outputs_eval["logits"].shape

    def test_different_sequence_lengths(self, sample_model, device):
        """Test model with different sequence lengths."""
        model = sample_model.to(device)

        # Test different sequence lengths
        for seq_len in [16, 32, 64]:
            batch = create_mock_batch(batch_size=2, seq_len=seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, return_dict=True)
            assert_tensor_shape(outputs["logits"], (2, 3), f"logits_seqlen_{seq_len}")

    def test_batch_size_flexibility(self, sample_model, device):
        """Test model with different batch sizes."""
        model = sample_model.to(device)

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            batch = create_mock_batch(batch_size=batch_size, seq_len=32)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, return_dict=True)
            assert_tensor_shape(outputs["logits"], (batch_size, 3), f"logits_batch_{batch_size}")

    def test_memory_efficiency(self, sample_model, device):
        """Test model memory usage."""
        model = sample_model.to(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)

            # Run forward pass
            batch = create_mock_batch(batch_size=2, seq_len=32)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, return_dict=True)
            peak_memory = torch.cuda.memory_allocated(device)

            # Memory should be reasonable
            memory_used = peak_memory - initial_memory
            assert memory_used > 0  # Should use some memory
            # Note: Specific memory limits depend on model size and are environment-dependent

    @pytest.mark.parametrize("num_adaptation_layers", [1, 2, 3])
    def test_different_adaptation_layers(self, genre_to_id_mapping, num_adaptation_layers, device):
        """Test model with different numbers of adaptation layers."""
        config = GenreAdaptiveNLIConfig(
            num_genres=len(genre_to_id_mapping),
            genre_adaptation_layers=num_adaptation_layers
        )
        model = GenreAdaptiveNLIValidator(config).to(device)

        batch = create_mock_batch(batch_size=2, seq_len=32)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, return_dict=True)
        assert_tensor_shape(outputs["logits"], (2, 3), f"logits_layers_{num_adaptation_layers}")

        # Check correct number of adaptation layers
        assert len(model.adaptation_layers) == num_adaptation_layers