import torch
import torch.nn as nn
from typing import Optional, Tuple

from model.blocks import DecoderBlock, EncoderBlock
from model.components import Embedder, KVCache
from model.config import TransformerConfig

class DecoderCache:
    """Per-layer KV caches plus the encoder output, reused across generation steps."""

    def __init__(self, num_layers: int):
        self.self_attn = [KVCache() for _ in range(num_layers)]
        self.cross_attn = [KVCache() for _ in range(num_layers)]
        self.encoder_output: Optional[torch.Tensor] = None
        self.encoder_padding_mask: Optional[torch.Tensor] = None
        self.self_padding_mask: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.self_attn[0])

    def extend_self_padding_mask(self, new_mask: torch.Tensor) -> torch.Tensor:
        if self.self_padding_mask is None:
            self.self_padding_mask = new_mask
        else:
            self.self_padding_mask = torch.cat((self.self_padding_mask, new_mask), dim=1)
        return self.self_padding_mask

    def reset(self):
        for cache in self.self_attn: cache.reset()
        for cache in self.cross_attn: cache.reset()
        self.encoder_output = None
        self.encoder_padding_mask = None
        self.self_padding_mask = None

class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.enc_vocab_dim, config.d_model, config.enc_max_seq_len,
                                 config.dropout, config.pad_id)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (encoder_output, non_padding_mask)."""
        curr_x, non_padding_mask = self.embedder(x)
        for enc_block in self.encoder_blocks:
            curr_x = enc_block(curr_x, non_padding_mask)
        return curr_x, non_padding_mask

class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.dec_vocab_dim, config.d_model, config.dec_max_seq_len,
                                 config.dropout, config.pad_id)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.linear = nn.Linear(config.d_model, config.dec_vocab_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, encoder_padding_mask: torch.Tensor,
                cache: Optional[DecoderCache] = None) -> torch.Tensor:
        """Returns logits of shape (batch, tgt_len, dec_vocab_dim)."""
        position_offset = len(cache) if cache is not None else 0
        embedded, non_padding_mask = self.embedder(x, position_offset=position_offset)

        if cache is not None:
            self_padding_mask = cache.extend_self_padding_mask(non_padding_mask)
        else:
            self_padding_mask = non_padding_mask

        curr_x = embedded
        for i, dec_block in enumerate(self.decoder_blocks):
            curr_x = dec_block(
                curr_x,
                self_padding_mask,
                encoder_output,
                encoder_padding_mask,
                self_cache=cache.self_attn[i] if cache is not None else None,
                cross_cache=cache.cross_attn[i] if cache is not None else None,
            )
        return self.linear(curr_x)

class Transformer(nn.Module):
    """Encoder-decoder Transformer (Vaswani et al., 2017)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self) -> DecoderCache:
        return DecoderCache(num_layers=self.config.n_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, cache: Optional[DecoderCache] = None) -> torch.Tensor:
        """x: source tokens (batch, src_len). y: target tokens (batch, tgt_len). Returns logits."""
        if cache is None or cache.encoder_output is None:
            encoder_output, encoder_padding_mask = self.encoder(x)
            if cache is not None:
                cache.encoder_output = encoder_output
                cache.encoder_padding_mask = encoder_padding_mask
        else:
            encoder_output = cache.encoder_output
            encoder_padding_mask = cache.encoder_padding_mask

        return self.decoder(y, encoder_output, encoder_padding_mask, cache=cache)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, bos_id: int, eos_id: int, max_new_tokens: int = 50,
                 use_cache: bool = True, on_token=None) -> torch.Tensor:
        """Greedy decoding. Sequences that emit eos_id are padded with pad_id afterwards."""
        was_training = self.training
        self.eval()

        batch_size = x.shape[0]
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        cache = self.new_cache() if use_cache else None

        for _ in range(max_new_tokens):
            decoder_input = generated[:, -1:] if use_cache else generated
            logits = self(x, decoder_input, cache=cache)
            next_token = logits[:, -1, :].argmax(dim=-1)
            
            next_token = next_token.masked_fill(finished, self.config.pad_id)
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)

            if on_token is not None:
                on_token(next_token.tolist())

            finished = finished | (next_token == eos_id)
            if bool(finished.all()):
                break

        if was_training:
            self.train()
        return generated
