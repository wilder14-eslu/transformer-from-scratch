from typing import Optional

import torch
import torch.nn as nn

from model.components import KVCache, MultiHeadAttention
from model.config import TransformerConfig


def feed_forward(config: TransformerConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.d_model, config.d_ff),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.d_ff, config.d_model),
        nn.Dropout(config.dropout),
    )


class EncoderBlock(nn.Module):
    """Self-attention + feed-forward, Post-LN as in the original paper."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads, d_model=config.d_model, d_head=config.d_head, dropout=config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = feed_forward(config)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    """Masked self-attention + cross-attention + feed-forward, Post-LN."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(
            num_heads=config.num_heads, d_model=config.d_model, d_head=config.d_head,
            masked=True, dropout=config.dropout,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=config.num_heads, d_model=config.d_model, d_head=config.d_head, dropout=config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.ffn = feed_forward(config)

    def forward(self, x: torch.Tensor, self_padding_mask: Optional[torch.Tensor],
                encoder_output: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor],
                self_cache: Optional[KVCache] = None, cross_cache: Optional[KVCache] = None) -> torch.Tensor:
        self_attn_out, _ = self.masked_self_attention(
            x, x, x, key_padding_mask=self_padding_mask, cache=self_cache
        )
        x = self.norm1(x + self_attn_out)

        # Queries come from the decoder; keys and values from the encoder.
        cross_attn_out, _ = self.cross_attention(
            x, encoder_output, encoder_output,
            key_padding_mask=encoder_padding_mask, cache=cross_cache, is_cross_attention=True,
        )
        x = self.norm2(x + cross_attn_out)

        x = self.norm3(x + self.ffn(x))
        return x
