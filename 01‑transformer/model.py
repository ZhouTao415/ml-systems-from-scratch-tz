import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding layer for transforming input token IDs into dense vectors.

    Attributes:
        d_model (int): Dimension of the embedding vectors (usually matches model hidden size).
        vocab_size (int): Total number of tokens in the vocabulary.
        embedding (nn.Embedding): Learnable embedding matrix of shape (vocab_size, d_model).
        scale (float): Scaling factor applied to embeddings to stabilize training (used in Transformer models).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Initialize embedding layer: maps token IDs to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Scale embeddings as recommended in "Attention Is All You Need"
        self.scale = math.sqrt(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Scaled embedding vectors of shape (batch_size, sequence_length, d_model).
        """
        return self.embedding(x) * self.scale
    
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in "Attention Is All You Need".
    Adds non-learnable position information to token embeddings to allow the model to utilize order.

    Attributes:
        d_model (int): Dimension of the embedding vectors.
        seq_len (int): Maximum sequence length supported.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        pe (Tensor): Precomputed positional encoding matrix (1, seq_len, d_model), not updated during training.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Initialize a positional encoding (pe) matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # pe(position, 2i) = sin(position / 10000^(2i/d_model))
        # pe(position, 2i+1) = cos(position / 10000^(2i/d_model))
        # Position index: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the scaling terms for the sinusoidal functions
        # Use exponential decay: 10000^(2i/d_model) => converted to exp form for efficiency and avoiding division
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension for broadcasting: (1, seq_len, d_model) 
        pe = pe.unsqueeze(0)

        # Register 'pe' as a persistent buffer (not trainable, but saved in model state) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Tensor: Output tensor with positional encoding added and dropout applied.
        """
        # Add positional encoding to input (no gradient needed for pe)
        x = x + self.pe[:, :x.size(1), :].detach()

        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Custom implementation of Layer Normalization.
    \hat{x}_j = (x_j - μ_j) / sqrt(σ_j² + ε)
    output = γ * \hat{x} + β
    Normalizes inputs across the last dimension and applies learnable scale (gamma) and bias (beta).

    Attributes:
        eps (float): Small constant added to the denominator for numerical stability.
        alpha (nn.Parameter): Learnable scaling factor (gamma), shape (1,).
        bias (nn.Parameter): Learnable bias term (beta), shape (1,).
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        # Initialize learnable parameters:
        # alpha (γ): scales normalized values
        # bias (β): shifts normalized values
        self.alpha = nn.Parameter(torch.ones(1))     # γ (scale)
        self.bias = nn.Parameter(torch.zeros(1))     # β (bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization on the last dimension of the input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., features)

        Returns:
            Tensor: Layer-normalized output with same shape as input.
        """
        # Compute mean and std over the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize and apply learnable affine transform
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network used in Transformer blocks.

    Consists of two linear transformations with a ReLU activation in between:
        FFN(x) = W2 * ReLU(W1 * x + b1) + b2

    Args:
        d_model (int): Input and output dimension of the model.
        d_ff (int): Hidden layer size in the feed-forward network (typically larger than d_model, e.g., 2048).
        dropout (float): Dropout probability applied after the activation.

    Attributes:
        linear1 (nn.Linear): First linear layer (d_model -> d_ff).
        dropout (nn.Dropout): Dropout layer.
        linear2 (nn.Linear): Second linear layer (d_ff -> d_model).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)      # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)      # W2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, seq_len, d_model)
        """
        # [Batch, seq_len, d_model] -> [Batch, seq_len, d_ff] -> [Batch, seq_len, d_model]
        # Step 1: Project to higher dimension
        # Step 2: Apply ReLU activation
        # Step 3: Apply dropout
        # Step 4: Project back to d_model
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
