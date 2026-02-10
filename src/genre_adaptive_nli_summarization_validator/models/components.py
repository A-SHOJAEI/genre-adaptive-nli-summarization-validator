"""Custom neural network components for genre-adaptive NLI model."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreConditionedGating(nn.Module):
    """Genre-conditioned gating mechanism for adaptive feature selection.

    This component learns to selectively gate features based on genre context,
    allowing the model to emphasize different linguistic patterns for different
    text domains (e.g., factual consistency in news vs. narrative coherence in fiction).
    """

    def __init__(
        self,
        hidden_size: int,
        genre_embedding_dim: int,
        dropout: float = 0.1
    ):
        """Initialize genre-conditioned gating.

        Args:
            hidden_size: Dimension of hidden states.
            genre_embedding_dim: Dimension of genre embeddings.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.genre_embedding_dim = genre_embedding_dim

        # Genre-to-gate transformation
        self.genre_gate_projection = nn.Sequential(
            nn.Linear(genre_embedding_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # Gate values in [0, 1]
        )

        # Context-aware gate modulation
        self.context_modulation = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        genre_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply genre-conditioned gating.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size].
            genre_embeddings: Genre embeddings [batch_size, genre_embedding_dim].

        Returns:
            Gated hidden states [batch_size, seq_len, hidden_size].
        """
        # Compute genre-based gates
        genre_gates = self.genre_gate_projection(genre_embeddings)  # [batch, hidden]
        genre_gates = genre_gates.unsqueeze(1)  # [batch, 1, hidden]

        # Context-aware modulation
        context_features = self.context_modulation(hidden_states)  # [batch, seq, hidden]

        # Apply gating: element-wise multiplication with broadcast
        gated_states = hidden_states * genre_gates + context_features * (1 - genre_gates)

        return gated_states


class CrossGenreAttention(nn.Module):
    """Cross-genre attention for learning genre-invariant entailment patterns.

    This component enables the model to attend across different genre representations,
    promoting transfer learning and consistent entailment semantics across domains.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize cross-genre attention.

        Args:
            hidden_size: Dimension of hidden states.
            num_attention_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        assert self.head_dim * num_attention_heads == hidden_size, \
            "hidden_size must be divisible by num_attention_heads"

        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden states into multiple attention heads.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size].

        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back to hidden states.

        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim].

        Returns:
            Merged tensor [batch_size, seq_len, hidden_size].
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-genre attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size].
            attention_mask: Optional attention mask [batch_size, seq_len].

        Returns:
            Attended hidden states [batch_size, seq_len, hidden_size].
        """
        residual = hidden_states

        # Compute Q, K, V
        query_states = self._split_heads(self.query(hidden_states))
        key_states = self._split_heads(self.key(hidden_states))
        value_states = self._split_heads(self.value(hidden_states))

        # Scaled dot-product attention
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads: [batch, 1, 1, seq_len]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + (1.0 - expanded_mask) * -10000.0

        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, value_states)
        context = self._merge_heads(context)

        # Output projection and residual connection
        output = self.output_projection(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class TemperatureScaledClassifier(nn.Module):
    """Temperature-scaled classification head for calibrated predictions.

    Uses learnable temperature scaling to improve confidence calibration,
    critical for summary validation where overconfident predictions can be costly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        initial_temperature: float = 1.0
    ):
        """Initialize temperature-scaled classifier.

        Args:
            hidden_size: Dimension of input features.
            num_labels: Number of output classes.
            dropout: Dropout probability.
            initial_temperature: Initial temperature value.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # Classification layers
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        apply_temperature: bool = True
    ) -> torch.Tensor:
        """Apply classification with temperature scaling.

        Args:
            hidden_states: Input features [batch_size, hidden_size].
            apply_temperature: Whether to apply temperature scaling.

        Returns:
            Logits [batch_size, num_labels].
        """
        # Dense transformation
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.dropout(x)

        # Classification logits
        logits = self.classifier(x)

        # Temperature scaling
        if apply_temperature:
            # Clamp temperature to prevent numerical issues
            temp = torch.clamp(self.temperature, min=0.1, max=10.0)
            logits = logits / temp

        return logits

    def get_calibrated_probabilities(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Get calibrated probability predictions.

        Args:
            hidden_states: Input features [batch_size, hidden_size].

        Returns:
            Calibrated probabilities [batch_size, num_labels].
        """
        logits = self.forward(hidden_states, apply_temperature=True)
        return F.softmax(logits, dim=-1)


class GenreAwarePooling(nn.Module):
    """Genre-aware pooling that adapts pooling strategy based on text genre.

    Different genres may require different information aggregation strategies
    (e.g., global context for news vs. local details for fiction).
    """

    def __init__(
        self,
        hidden_size: int,
        genre_embedding_dim: int,
        pooling_strategies: int = 3
    ):
        """Initialize genre-aware pooling.

        Args:
            hidden_size: Dimension of hidden states.
            genre_embedding_dim: Dimension of genre embeddings.
            pooling_strategies: Number of pooling strategies to combine.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.genre_embedding_dim = genre_embedding_dim
        self.pooling_strategies = pooling_strategies

        # Genre-to-pooling weight transformation
        self.pooling_weight_net = nn.Sequential(
            nn.Linear(genre_embedding_dim, pooling_strategies * 2),
            nn.GELU(),
            nn.Linear(pooling_strategies * 2, pooling_strategies),
            nn.Softmax(dim=-1)
        )

        # Attention-based pooling
        self.attention_pooling = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        genre_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply genre-aware pooling.

        Args:
            hidden_states: Sequence hidden states [batch_size, seq_len, hidden_size].
            genre_embeddings: Genre embeddings [batch_size, genre_embedding_dim].
            attention_mask: Optional mask [batch_size, seq_len].

        Returns:
            Pooled representation [batch_size, hidden_size].
        """
        # Strategy 1: Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
        else:
            mean_pooled = hidden_states.mean(dim=1)

        # Strategy 2: Max pooling
        if attention_mask is not None:
            masked_states = hidden_states.clone()
            masked_states[~attention_mask.unsqueeze(-1).expand_as(hidden_states)] = -1e9
            max_pooled = masked_states.max(dim=1)[0]
        else:
            max_pooled = hidden_states.max(dim=1)[0]

        # Strategy 3: Attention pooling
        attention_scores = self.attention_pooling(hidden_states)  # [batch, seq, 1]
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.unsqueeze(-1), -1e9
            )
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_pooled = (hidden_states * attention_weights).sum(dim=1)

        # Stack pooling strategies
        pooled_strategies = torch.stack([
            mean_pooled,
            max_pooled,
            attention_pooled
        ], dim=1)  # [batch, strategies, hidden]

        # Get genre-specific pooling weights
        pooling_weights = self.pooling_weight_net(genre_embeddings)  # [batch, strategies]
        pooling_weights = pooling_weights.unsqueeze(-1)  # [batch, strategies, 1]

        # Weighted combination of pooling strategies
        final_pooled = (pooled_strategies * pooling_weights).sum(dim=1)  # [batch, hidden]

        return final_pooled
