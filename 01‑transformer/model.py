import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

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

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention block from the Transformer architecture.

    Args:
        d_model (int): Dimension of the input embeddings (must be divisible by n_heads).
        n_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability applied to attention weights.

    Attributes:
        w_q, w_k, w_v (nn.Linear): Linear projections for query, key, value.
        w_o (nn.Linear): Output linear projection after concatenating all heads.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod  # This method does not depend on the class instance; it can be called as MultiHeadAttentionBlock.attention_score(...)
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        dropout: Optional[nn.Dropout]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query (Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            key (Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            value (Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_v)
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, 1, seq_len) or similar, with 0 for masked positions
            dropout (nn.Dropout, optional): Dropout module applied on attention weights

        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor after applying attention, shape (batch_size, n_heads, seq_len, d_v)
                - Attention weights (for visualization), shape (batch_size, n_heads, seq_len, seq_len)
        """
        d_k = query.shape[-1]  # Key/query dimension per head

        # Compute raw attention scores: (Q • K^T) / sqrt(d_k)
        # (B, h, L_q, d_k) @ (B, h, d_k, L_k) -> (B, h, L_q, L_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask: fill masked positions with large negative number so softmax ~ 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Normalize scores to probabilities
        attention_weights = attention_scores.softmax(dim=-1)

        # Optionally apply dropout to attention weights
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        # Weighted sum of values: (B, h, L_q, L_k) @ (B, h, L_k, d_v) -> (B, h, L_q, d_v)
        output = attention_weights @ value

        return output, attention_weights

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply multi-head attention to the input.

        Args:
            q, k, v (Tensor): Input tensors of shape (Batch, seq_len, d_model)
            mask (Tensor, optional): Mask tensor
                if we want some words not to interact with others, we mask them
                and with very small value, exp with softmax will be 0

        Returns:
            Tensor: Output of shape (Batch, seq_len, d_model)
        """
        B, L, _ = q.size()
        # Linear projections: (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)
        
        # Reshape and transpose for multi-head attention:
        # (Batch, seq_len, d_model) → (Batch, seq_len, n_heads, d_k) → (Batch, n_heads, seq_len, d_k)
        # Transpose to move n_heads forward so that each attention head attends to the full sequence (seq_len)
        # Each head processes the entire sentence, but focuses on a different subspace of the embedding (d_k)
        # query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2) 
        # key   = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        # value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        query = query.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        x, attentions_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, n_heads, seq_len, d_k) → (Batch, seq_len, n_heads, d_k) → (Batch, seq_len, d_model)
        # Transpose to bring seq_len next to batch for output shape
        # Use .contiguous() to ensure the tensor is stored in contiguous memory before reshaping
        # -1 lets PyTorch automatically infer the seq_len dimension during .view()

        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization:
        x -> LayerNorm(x) -> Sublayer -> Dropout -> Add(x)

    This is the Pre-Norm variant used in modern Transformer implementations.

    Args:
        dropout (float): Dropout probability applied after the sublayer.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same input/output shape.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            sublayer (Callable): A function/layer taking and returning a tensor of the same shape.

        Returns:
            Tensor: Output tensor after applying norm -> sublayer -> dropout -> residual add
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    A single Transformer encoder block consisting of:
    - Multi-head self-attention layer
    - Feed-forward network (FFN)
    - Two residual connections with layer normalization (Pre-Norm)
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First residual: Self-attention
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # Second residual: Feed-forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder: stack of N encoder blocks with a final layer normalization.

    Args:
        layers (nn.ModuleList): List of EncoderBlock modules
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block consisting of:
    - Masked Multi-head Self-Attention (with causal mask)
    - Cross Multi-head Attention over encoder outputs
    - Feed Forward Network (FFN)
    - Each sublayer is wrapped in a ResidualConnection (Pre-Norm style)
    # src_mask is used to mask the encoder outputs, tgt_mask is used to mask the decoder inputs
    # tgt_mask is typically a causal mask to prevent attending to future tokens in the sequence
    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Masked self-attention (causal mask)
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )

        # Cross-attention over encoder outputs
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_outputs, encoder_outputs, src_mask)
        )

        # Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder composed of a stack of DecoderBlocks and a final layer normalization.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Projects decoder output embeddings into vocabulary logits for prediction.

    Args:
        d_model (int): Dimensionality of decoder outputs.
        vocab_size (int): Number of tokens in the vocabulary.

    Attributes:
        proj (nn.Linear): Linear layer projecting from d_model to vocab_size.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns log-probabilities over the vocabulary.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Log-probabilities over the vocabulary,
                    shape (batch_size, seq_len, vocab_size)
        """
        logits = self.proj(x)
        return torch.log_softmax(logits, dim=-1)
