# Transformer From Scratch

A clean, readable PyTorch implementation of the encoder-decoder Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
written from first principles without `nn.Transformer` or `nn.MultiheadAttention`.

> **Reference implementation.** This repository focuses on the architecture. It does not
> include a training pipeline or pretrained weights. Correctness is checked with
> lightweight unit tests that run on CPU in a few seconds.

## Architecture

```
src tokens ──► Embedder ──► [EncoderBlock × N] ──────────────┐
                            self-attention + FFN             │ encoder output
                                                             ▼
tgt tokens ──► Embedder ──► [DecoderBlock × N] ──► Linear ──► logits
                            masked self-attention
                            cross-attention (Q: decoder, K/V: encoder)
                            FFN
```

| Component | Details |
|---|---|
| Embeddings | Token embeddings + learned positional embeddings |
| Attention | Multi-head scaled dot-product attention, written explicitly |
| Masks | Key padding mask (encoder and decoder) and causal mask (decoder) |
| Normalization | Post-LN, as in the original paper |
| Feed-forward | Linear → GELU → Dropout → Linear |
| Initialization | Normal(0, 0.02) for linear and embedding weights |
| Inference | Greedy decoding with a per-layer **KV cache** |

With the default `TransformerConfig` (the paper's "base" sizes, 10k vocabularies), the model
has about **60M parameters**.

### KV cache

During autoregressive generation, `generate()` keeps a `DecoderCache` with:

- **Self-attention cache**: keys and values of already generated tokens, extended by one position per step.
- **Cross-attention cache**: the encoder output projected to keys and values once and reused at every step.
- **Encoder output**: the source is encoded only once.

The causal mask accounts for cached positions, so each new query attends to every past token
without recomputing them. A unit test checks that cached and uncached generation produce
identical outputs.

## Project structure

```
model/
├── config.py        TransformerConfig dataclass
├── components.py    KVCache, Embedder, MultiHeadAttention
├── blocks.py        EncoderBlock, DecoderBlock
└── transformer.py   Encoder, Decoder, DecoderCache, Transformer (+ generate)
tests/
└── test_model.py    Architecture sanity checks
```

## Usage

```python
import torch
from model import Transformer, TransformerConfig

config = TransformerConfig(d_model=128, d_head=32, d_ff=512, num_heads=4, n_layers=2,
                           enc_vocab_dim=1000, dec_vocab_dim=1000)
model = Transformer(config)

src = torch.randint(3, 1000, (2, 10))        # (batch, src_len), 0 = padding
tgt = torch.randint(3, 1000, (2, 8))         # (batch, tgt_len)
logits = model(src, tgt)                     # (2, 8, 1000)

generated = model.generate(src, bos_id=1, eos_id=2, max_new_tokens=20)
```

## Tests

```bash
pip install -r requirements.txt
pytest
```

The tests verify:

- Output shapes and finite values
- Causality: changing a future target token does not affect earlier positions
- Source padding does not change the output
- Generation with and without KV cache is identical
- Gradients reach every parameter
- Masking works in reduced precision (fp16/bf16)

## Possible extensions

- Pre-LN blocks for more stable training of deeper models
- Sinusoidal or rotary positional encodings
- `F.scaled_dot_product_attention` for fused (Flash) attention kernels
- Beam search and sampling strategies (top-k, top-p, temperature)
- A decoder-only (GPT-style) variant

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
*Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
