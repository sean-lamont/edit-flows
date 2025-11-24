from typing import Tuple, cast
from torchtyping import TensorType as T

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Simple sinusoidal time embedding for Transformer model
    """

    def __init__(self, hidden_dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: T["batch", 1, "float"]) -> T["batch", "hidden_dim"]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Make it (batch_size, 1)

        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)  # Broadcasting: (batch_size, 1) * (1, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd hidden_dim
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb  # (batch_size, hidden_dim)


class SimpleEditFlowsTransformer(nn.Module):
    """
    Small vanilla Transformer model for edit flows with padding support.
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            num_heads=8,
            max_seq_len=512,
            bos_token_id=128,
            pad_token_id=129,
    ):
        super(SimpleEditFlowsTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        assert bos_token_id < vocab_size, "bos_token_id must be less than vocab_size"
        assert pad_token_id < vocab_size, "pad_token_id must be less than vocab_size"

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)  # Token embeddings
        self.time_embedding = nn.Sequential(  # Time embeddings
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)  # Positional embeddings
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                       dim_feedforward=hidden_dim * 4,
                                       dropout=0.1, activation='gelu',
                                       batch_first=False)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Output 3 rates (insert, substitute, delete)
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size insert probabilities
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),  # Output vocab_size substitute probabilities
        )
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: T["batch", "x_seq_len", "long"],
                time_step: T["batch", 1, "float"],
                padding_mask: T["batch", "x_seq_len", "bool"]) \
            -> Tuple[
                T["batch", "x_seq_len", "float"],  # Rates (3 values)
                T["batch", "x_seq_len", "vocab_size"],  # Insert probabilities (vocab_size values)
                T["batch", "x_seq_len", "vocab_size"],  # Substitute probabilities (vocab_size values)
            ]:
        """Forward pass takes in x_t, t, and padding mask, returns rates and probabilities
        """
        batch_size, x_seq_len = tokens.shape
        token_emb = self.token_embedding(tokens)  # (batch_size, x_seq_len, hidden_dim)

        time_emb = self.time_embedding(time_step)  # (batch_size, hidden_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, x_seq_len, -1)  # (batch_size, x_seq_len, hidden_dim)

        positions = torch.arange(x_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)  # (batch_size, x_seq_len, hidden_dim)

        x = token_emb + time_emb + pos_emb  # (batch_size, x_seq_len, hidden_dim)
        x = x.transpose(0, 1)  # expects (x_seq_len, batch_size, hidden_dim)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = x.transpose(0, 1)  # (batch_size, x_seq_len, hidden_dim)
        x = self.final_layer_norm(x)  # (batch_size, x_seq_len, hidden_dim)
        ins_logits = self.ins_logits_out(x)  # (batch_size, x_seq_len, vocab_size)
        sub_logits = self.sub_logits_out(x)  # (batch_size, x_seq_len, vocab_size)
        rates = F.softplus(self.rates_out(x))  # (batch_size, x_seq_len, 3) - ensure positive rates

        ins_probs = F.softmax(ins_logits, dim=-1)  # (batch_size, x_seq_len, vocab_size)
        sub_probs = F.softmax(sub_logits, dim=-1)  # (batch_size, x_seq_len, vocab_size)

        # Zero out outputs for padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, x_seq_len, 1)
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded

        if torch.isnan(rates).any() or torch.isnan(ins_probs).any() or torch.isnan(sub_probs).any():
            raise ValueError("NaN detected in output probabilities or rates")

        return (
            cast(T["batch", "x_seq_len", "float"], rates),
            cast(T["batch", "x_seq_len", "vocab_size"], ins_probs),
            cast(T["batch", "x_seq_len", "vocab_size"], sub_probs),
        )
