"""Genre-adaptive NLI model for summary validation."""

import logging
from typing import Dict, Optional, Tuple, Union, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig
)
import numpy as np

from ..exceptions import ModelError


class GenreAdaptiveNLIConfig(PretrainedConfig):
    """Configuration class for GenreAdaptiveNLIValidator."""

    model_type = "genre_adaptive_nli"

    def __init__(
        self,
        base_model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        num_genres: int = 10,
        genre_embedding_dim: int = 128,
        genre_adaptation_layers: int = 2,
        dropout: float = 0.1,
        genre_attention_heads: int = 8,
        cross_genre_regularization: float = 0.1,
        temperature_scaling: bool = True,
        **kwargs
    ):
        """Initialize configuration.

        Args:
            base_model_name: Name of base transformer model.
            num_labels: Number of NLI labels (entailment, neutral, contradiction).
            num_genres: Number of genre categories.
            genre_embedding_dim: Dimension of genre embeddings.
            genre_adaptation_layers: Number of genre adaptation layers.
            dropout: Dropout probability.
            genre_attention_heads: Number of attention heads for genre adaptation.
            cross_genre_regularization: Weight for cross-genre regularization loss.
            temperature_scaling: Whether to use temperature scaling for calibration.
        """
        super().__init__(**kwargs)

        self.encoder_name = base_model_name
        self.num_labels = num_labels
        self.num_genres = num_genres
        self.genre_embedding_dim = genre_embedding_dim
        self.genre_adaptation_layers = genre_adaptation_layers
        self.dropout = dropout
        self.genre_attention_heads = genre_attention_heads
        self.cross_genre_regularization = cross_genre_regularization
        self.temperature_scaling = temperature_scaling


class GenreAdaptationLayer(nn.Module):
    """Genre-specific adaptation layer with multi-head attention."""

    def __init__(
        self,
        hidden_size: int,
        genre_embedding_dim: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize genre adaptation layer.

        Args:
            hidden_size: Hidden size of the base model.
            genre_embedding_dim: Dimension of genre embeddings.
            num_attention_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.genre_embedding_dim = genre_embedding_dim
        self.num_attention_heads = num_attention_heads

        # Genre-conditioned attention
        self.genre_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Genre-specific transformations
        self.genre_projection = nn.Linear(genre_embedding_dim, hidden_size)
        self.adaptation_gate = nn.Linear(hidden_size + genre_embedding_dim, hidden_size)

        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        genre_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through genre adaptation layer.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size].
            genre_embeddings: Genre embeddings [batch_size, genre_embedding_dim].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            Adapted hidden states.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project genre embeddings to hidden space
        genre_proj = self.genre_projection(genre_embeddings)  # [batch_size, hidden_size]
        genre_proj = genre_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]

        # Genre-conditioned self-attention
        query = hidden_states + genre_proj
        attn_output, _ = self.genre_attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )

        # Residual connection and layer norm
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attn_output))

        # Genre-specific gating
        gate_input = torch.cat([
            hidden_states,
            genre_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        ], dim=-1)
        gate = torch.sigmoid(self.adaptation_gate(gate_input))
        hidden_states = hidden_states * gate

        # Feedforward network
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ffn_output)

        return hidden_states


class GenreAdaptiveNLIValidator(PreTrainedModel):
    """Genre-adaptive NLI model for summarization validation.

    This model combines a transformer base with genre-specific adaptation layers
    to improve entailment detection across different text genres.
    """

    config_class = GenreAdaptiveNLIConfig

    def __init__(self, config: GenreAdaptiveNLIConfig):
        """Initialize genre-adaptive NLI validator.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        self.config = config
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initializing GenreAdaptiveNLIValidator with {config.num_genres} genres")
        self.logger.debug(f"Model configuration: {config.to_dict()}")

        # Base transformer model
        try:
            self.logger.info(f"Loading base model: {config.encoder_name}")
            base_config = AutoConfig.from_pretrained(config.encoder_name)
            self.encoder = AutoModel.from_pretrained(
                config.encoder_name, config=base_config, torch_dtype=torch.float32
            )
            self.hidden_size = base_config.hidden_size
            self.logger.info(f"Base model loaded successfully. Hidden size: {self.hidden_size}")
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise ModelError(
                f"Failed to load base model '{config.encoder_name}'",
                details={
                    "base_model_name": config.encoder_name,
                    "num_genres": config.num_genres
                },
                original_exception=e
            )

        # Genre embeddings
        self.genre_embeddings = nn.Embedding(
            config.num_genres,
            config.genre_embedding_dim
        )

        # Genre adaptation layers
        self.adaptation_layers = nn.ModuleList([
            GenreAdaptationLayer(
                hidden_size=self.hidden_size,
                genre_embedding_dim=config.genre_embedding_dim,
                num_attention_heads=config.genre_attention_heads,
                dropout=config.dropout
            )
            for _ in range(config.genre_adaptation_layers)
        ])

        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        # Temperature scaling for calibration
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter('temperature', None)

        # Genre-specific bias terms
        self.genre_bias = nn.Embedding(config.num_genres, config.num_labels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize genre-adaptive model weights using standard initialization schemes.

        Initializes all custom model components with appropriate distributions:
        - Genre embeddings: Normal distribution with std=0.02
        - Classification head: Normal weight initialization, zero bias
        - Genre bias: Zero initialization for neutral starting point
        - Temperature parameter: Unit initialization for calibration

        Note:
            This method is called automatically during model initialization and
            does not initialize the pre-trained transformer weights, which retain
            their original initialization from the base model.
        """
        # Initialize genre embeddings
        nn.init.normal_(self.genre_embeddings.weight, std=0.02)

        # Initialize classification head
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Initialize genre bias
        nn.init.zeros_(self.genre_bias.weight)

        # Initialize temperature
        if self.temperature is not None:
            nn.init.ones_(self.temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        genre_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            token_type_ids: Token type IDs [batch_size, seq_len].
            genre_ids: Genre IDs [batch_size].
            labels: Ground truth labels [batch_size].
            return_dict: Whether to return a dictionary.

        Returns:
            Model outputs including loss and logits.
        """
        # Encode with base model
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Apply genre adaptation if genre IDs provided
        if genre_ids is not None:
            genre_emb = self.genre_embeddings(genre_ids)  # [batch_size, genre_embedding_dim]

            # Apply genre adaptation layers
            for adaptation_layer in self.adaptation_layers:
                hidden_states = adaptation_layer(
                    hidden_states=hidden_states,
                    genre_embeddings=genre_emb,
                    attention_mask=attention_mask
                )

        # Pool sequence representation (CLS token)
        pooled_output = hidden_states[:, 0]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        # Add genre-specific bias
        if genre_ids is not None:
            genre_bias = self.genre_bias(genre_ids)  # [batch_size, num_labels]
            logits = logits + genre_bias

        # Apply temperature scaling
        if self.temperature is not None:
            logits = logits / self.temperature

        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            # Add cross-genre regularization if genre IDs provided
            if genre_ids is not None and self.config.cross_genre_regularization > 0:
                reg_loss = self._compute_cross_genre_regularization(genre_ids, labels, logits)
                loss = loss + self.config.cross_genre_regularization * reg_loss

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states,
                "pooled_output": pooled_output
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs

    def _compute_cross_genre_regularization(
        self,
        genre_ids: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-genre regularization loss.

        Encourages similar predictions for similar examples across genres.

        Args:
            genre_ids: Genre IDs [batch_size].
            labels: Ground truth labels [batch_size].
            logits: Model logits [batch_size, num_labels].

        Returns:
            Regularization loss.
        """
        batch_size = genre_ids.size(0)

        # Compute pairwise label similarity
        label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Compute pairwise genre difference
        genre_diff = (genre_ids.unsqueeze(0) != genre_ids.unsqueeze(1)).float()

        # Compute prediction similarity (cosine similarity)
        norm_logits = F.normalize(logits, p=2, dim=1)
        pred_sim = torch.mm(norm_logits, norm_logits.t())

        # Regularization: similar labels across different genres should have similar predictions
        reg_mask = label_sim * genre_diff
        reg_loss = torch.sum(reg_mask * (1 - pred_sim)) / (torch.sum(reg_mask) + 1e-8)

        return reg_loss

    def predict_entailment_score(
        self,
        premise: str,
        hypothesis: str,
        genre: str,
        tokenizer: AutoTokenizer,
        genre_to_id: Dict[str, int],
        max_length: int = 512,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Predict entailment score for premise-hypothesis pair.

        Args:
            premise: Premise text.
            hypothesis: Hypothesis text.
            genre: Text genre.
            tokenizer: Tokenizer for text encoding.
            genre_to_id: Mapping from genre names to IDs.
            max_length: Maximum sequence length.
            device: Device to run inference on.

        Returns:
            Dictionary with prediction scores and metadata.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Tokenize inputs
        inputs = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # Get genre ID
        genre_id = genre_to_id.get(genre, 0)  # Default to first genre if unknown
        genre_tensor = torch.tensor([genre_id], device=device)

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                genre_ids=genre_tensor,
                return_dict=True
            )

        # Get predictions
        logits = outputs["logits"].squeeze(0)  # [num_labels]
        probs = F.softmax(logits, dim=0)

        # Compute entailment score
        entailment_score = probs[0].item()  # Entailment probability
        neutral_score = probs[1].item()
        contradiction_score = probs[2].item()

        # Compute confidence
        confidence = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

        return {
            "entailment_score": entailment_score,
            "neutral_score": neutral_score,
            "contradiction_score": contradiction_score,
            "confidence": confidence,
            "entropy": entropy,
            "predicted_label": torch.argmax(probs).item(),
            "genre": genre
        }

    def validate_summary(
        self,
        document: str,
        summary: str,
        genre: str,
        tokenizer: AutoTokenizer,
        genre_to_id: Dict[str, int],
        max_length: int = 512,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Validate summary against source document.

        Args:
            document: Source document.
            summary: Summary to validate.
            genre: Document genre.
            tokenizer: Tokenizer instance.
            genre_to_id: Genre to ID mapping.
            max_length: Maximum sequence length.
            threshold: Entailment threshold for validation.

        Returns:
            Validation results including scores and decision.
        """
        # Get entailment prediction
        result = self.predict_entailment_score(
            premise=document,
            hypothesis=summary,
            genre=genre,
            tokenizer=tokenizer,
            genre_to_id=genre_to_id,
            max_length=max_length
        )

        # Make validation decision
        is_valid = result["entailment_score"] >= threshold

        # Add validation metadata
        result.update({
            "is_valid": is_valid,
            "threshold_used": threshold,
            "summary_length": len(summary.split()),
            "document_length": len(document.split()),
            "compression_ratio": len(summary.split()) / len(document.split())
        })

        return result

    @classmethod
    def from_pretrained_with_config(
        cls,
        model_name_or_path: str,
        config: Optional[GenreAdaptiveNLIConfig] = None,
        **kwargs
    ) -> "GenreAdaptiveNLIValidator":
        """Load pre-trained model with custom configuration.

        Args:
            model_name_or_path: Path or name of pre-trained model.
            config: Custom configuration.
            **kwargs: Additional arguments.

        Returns:
            Loaded model instance.
        """
        if config is None:
            config = GenreAdaptiveNLIConfig.from_pretrained(model_name_or_path)

        model = cls(config)

        # Load pre-trained weights if available
        try:
            model.load_state_dict(
                torch.load(f"{model_name_or_path}/pytorch_model.bin", map_location="cpu")
            )
        except FileNotFoundError:
            logging.warning(f"No pre-trained weights found at {model_name_or_path}")

        return model