"""
AirType LSTM Model Architecture

This module defines the bidirectional LSTM model with attention mechanism
for handwriting character recognition.
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention weights using scaled dot-product of query and key,
    then applies these weights to the values.
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            dim: Dimension of the attention space
            dropout: Dropout probability
        """
        super().__init__()
        self.scale = math.sqrt(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention output.
        
        Args:
            query: Query tensor (batch, seq_len, dim)
            key: Key tensor (batch, seq_len, dim)
            value: Value tensor (batch, seq_len, dim)
            mask: Optional attention mask (batch, seq_len)
        
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting: (batch, 1, seq_len)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.
    
    Projects inputs into multiple attention heads, computes attention
    in parallel, then concatenates and projects back.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Input/output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Attention
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask (batch, seq_len)
        
        Returns:
            Tuple of (output, attention weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention for each head
        attn_output, attn_weights = self.attention(query, key, value, mask)
        
        # Transpose and reshape back: (batch, seq_len, dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


class HandwritingLSTM(nn.Module):
    """
    Bidirectional LSTM model with attention for handwriting recognition.
    
    Architecture:
    - Input embedding: 7 features -> 64 dimensions
    - LSTM Layer 1: 64 hidden units, bidirectional
    - LSTM Layer 2: 128 hidden units, bidirectional
    - Multi-head attention over sequence
    - FC layers: 256 -> 128 -> 62 classes
    
    Output classes: 26 lowercase + 26 uppercase + 10 digits = 62 characters
    """
    
    # Character mappings
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS)
    
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 64,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 128,
        num_attention_heads: int = 4,
        fc_dim: int = 256,
        num_classes: int = 62,
        dropout: float = 0.3,
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features per point (x, y, vx, vy, acc, curv, pen)
            embed_dim: Embedding dimension
            hidden_dim_1: Hidden dimension for first LSTM layer
            hidden_dim_2: Hidden dimension for second LSTM layer
            num_attention_heads: Number of attention heads
            fc_dim: Dimension of fully connected layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_classes = num_classes
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,  # No dropout between layers, using explicit dropout
        )
        
        self.lstm_dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim_1 * 2,  # Bidirectional output
            hidden_size=hidden_dim_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        
        self.lstm_dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism
        attention_dim = hidden_dim_2 * 2  # Bidirectional output
        self.attention = MultiHeadAttention(
            dim=attention_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        
        self.attention_norm = nn.LayerNorm(attention_dim)
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, fc_dim),
            nn.LayerNorm(fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.LayerNorm(fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    # LSTM weights: orthogonal initialization
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                elif "embedding" in name or "classifier" in name:
                    if len(param.shape) >= 2:
                        nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            lengths: Sequence lengths for packing (batch,)
            mask: Attention mask (batch, seq_len), 1 for valid, 0 for padding
        
        Returns:
            Tuple of (logits, attention_weights)
            - logits: (batch, num_classes)
            - attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM Layer 1
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm1_out, _ = self.lstm1(packed)
            lstm1_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm1_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm1_out, _ = self.lstm1(embedded)
        
        lstm1_out = self.lstm_dropout1(lstm1_out)
        
        # LSTM Layer 2
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                lstm1_out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm2_out, _ = self.lstm2(packed)
            lstm2_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm2_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm2_out, _ = self.lstm2(lstm1_out)
        
        lstm2_out = self.lstm_dropout2(lstm2_out)
        
        # Attention
        attn_output, attn_weights = self.attention(lstm2_out, mask)
        
        # Residual connection and layer norm
        attn_output = self.attention_norm(attn_output + lstm2_out)
        
        # Global average pooling over sequence
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (attn_output * mask_expanded).sum(dim=1)
            pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = attn_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attn_weights
    
    def predict(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> dict:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            lengths: Sequence lengths (batch,)
            mask: Attention mask (batch, seq_len)
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(x, lengths, mask)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=-1)
            
            results = []
            for i in range(x.shape[0]):
                predicted_idx = top_indices[i, 0].item()
                predicted_char = self.IDX_TO_CHAR.get(predicted_idx, "?")
                confidence = top_probs[i, 0].item()
                
                alternatives = []
                for j in range(1, min(top_k, self.num_classes)):
                    alt_idx = top_indices[i, j].item()
                    alt_char = self.IDX_TO_CHAR.get(alt_idx, "?")
                    alt_conf = top_probs[i, j].item()
                    alternatives.append({
                        "char": alt_char,
                        "confidence": alt_conf,
                    })
                
                results.append({
                    "predicted_char": predicted_char,
                    "confidence": confidence,
                    "alternatives": alternatives,
                })
            
            return results
    
    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "HandwritingLSTM":
        """
        Load a pretrained model from disk.
        
        Args:
            path: Path to the model checkpoint
            device: Device to load the model on
        
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Handle both full checkpoint and state_dict only
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            model = cls(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = cls()
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        return model
    
    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer to save
            epoch: Current epoch number
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "embed_dim": self.embed_dim,
                "hidden_dim_1": self.hidden_dim_1,
                "hidden_dim_2": self.hidden_dim_2,
                "num_classes": self.num_classes,
            },
            "epoch": epoch,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Smooths the target distribution to prevent overconfidence
    and improve generalization.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = "mean"):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0-1)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            pred: Predictions (batch, num_classes)
            target: Target indices (batch,)
        
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed target distribution
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute cross-entropy with smoothed targets
        loss = -(smooth_target * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
