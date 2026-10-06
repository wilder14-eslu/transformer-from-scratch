import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class KVCache:
    """Stores keys and values of past positions for one attention layer."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None or self.v is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=-2)
            self.v = torch.cat((self.v, v), dim=-2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


class Embedder(nn.Module):
    """Token embeddings + learned positional embeddings."""

    def __init__(self, vocab_dim: int, embedding_dim: int, max_seq_len: int,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.embeddings = nn.Embedding(vocab_dim, embedding_dim, padding_idx=pad_id)
        self.pos_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeddings, non_padding_mask). The mask is True for real tokens."""
        seq_len = x.shape[1]
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {position_offset + seq_len} exceeds maximum {self.max_seq_len}")

        positions = torch.arange(position_offset, position_offset + seq_len, device=x.device)
        non_padding_mask = x != self.pad_id

        embeddings = self.dropout(self.embeddings(x) + self.pos_embeddings(positions))
        return embeddings, non_padding_mask


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with optional causal mask and KV cache."""

    def __init__(self, num_heads: int, d_model: int, d_head: int, masked: bool = False, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_head
        self.masked = masked

        self.W_q = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.W_o = nn.Linear(d_head * num_heads, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, heads * d_head) -> (batch, heads, seq, d_head)
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, heads, seq, d_head) -> (batch, seq, heads * d_head)
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.d_head)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None,
                is_cross_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch, q_len, d_model)
        key, value: (batch, kv_len, d_model)
        key_padding_mask: (batch, kv_len), True for real tokens.
        Returns (output, attention_weights).
        """
        Q = self._split_heads(self.W_q(query))

        if is_cross_attention and cache is not None and len(cache) > 0:
            # Encoder output does not change during generation: reuse projected K, V.
            assert cache.k is not None and cache.v is not None
            K, V = cache.k, cache.v
        else:
            K = self._split_heads(self.W_k(key))
            V = self._split_heads(self.W_v(value))
            if cache is not None:
                K, V = cache.update(K, V)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # dtype-aware minimum: a hardcoded -1e9 overflows in fp16 under autocast.
        mask_value = torch.finfo(scores.dtype).min

        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], mask_value)

        if self.masked:
            # With a cache, the queries are the last q_len positions of a kv_len sequence.
            q_len, kv_len = Q.shape[-2], K.shape[-2]
            past_len = kv_len - q_len
            q_positions = torch.arange(past_len, kv_len, device=Q.device).unsqueeze(-1)
            k_positions = torch.arange(kv_len, device=Q.device)
            scores = scores.masked_fill(q_positions < k_positions, mask_value)

        attention_weights = self.dropout(torch.softmax(scores, dim=-1))
        output = self._merge_heads(attention_weights @ V)
        return self.W_o(output), attention_weights
