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
