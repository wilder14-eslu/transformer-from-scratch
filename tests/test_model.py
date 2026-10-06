"""Architecture sanity checks. No training: each test runs on CPU in well under a second."""

import torch

from model import Transformer, TransformerConfig

BOS, EOS = 1, 2


def small_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    config = TransformerConfig(
        d_model=32, d_head=8, d_ff=64, num_heads=4, n_layers=2,
        enc_max_seq_len=32, dec_max_seq_len=32,
        enc_vocab_dim=20, dec_vocab_dim=25, dropout=0.0,
    )
    return Transformer(config).eval()


def test_forward_output_shape():
    model = small_model()
    src = torch.randint(3, 20, (2, 7))
    tgt = torch.randint(3, 25, (2, 5))
    logits = model(src, tgt)
    assert logits.shape == (2, 5, 25)
    assert torch.isfinite(logits).all()


def test_decoder_is_causal():
    """Changing a future target token must not change logits at earlier positions."""
    model = small_model()
    src = torch.randint(3, 20, (1, 6))
    tgt = torch.randint(3, 25, (1, 6))
    tgt_changed = tgt.clone()
    tgt_changed[0, -1] = (tgt[0, -1] + 1) % 22 + 3

    out_a = model(src, tgt)
    out_b = model(src, tgt_changed)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


def test_source_padding_is_ignored():
    """Appending padding to the source must not change the output."""
    model = small_model()
    src = torch.randint(3, 20, (1, 5))
    src_padded = torch.cat([src, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    tgt = torch.randint(3, 25, (1, 4))
    assert torch.allclose(model(src, tgt), model(src_padded, tgt), atol=1e-5)


def test_generate_with_and_without_cache_match():
    """KV cache is an optimization only: greedy outputs must be identical."""
    model = small_model()
    src = torch.randint(3, 20, (3, 6))
    with_cache = model.generate(src, bos_id=BOS, eos_id=EOS, max_new_tokens=12, use_cache=True)
    without_cache = model.generate(src, bos_id=BOS, eos_id=EOS, max_new_tokens=12, use_cache=False)
    assert torch.equal(with_cache, without_cache)


def test_backward_reaches_all_parameters():
    model = small_model().train()
    src = torch.randint(3, 20, (2, 6))
    tgt = torch.randint(3, 25, (2, 5))
    model(src, tgt).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_fp16_masking_does_not_overflow():
    """Masking must work in half precision (regression test for the old -1e9 mask value)."""
    model = small_model()
    src = torch.tensor([[5, 6, 7, 0, 0]])
    tgt = torch.tensor([[BOS, 4, 5]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(src, tgt)
    assert torch.isfinite(logits.float()).all()
    if torch.cuda.is_available():
        model = model.cuda().half()
        logits = model(src.cuda(), tgt.cuda())
        assert torch.isfinite(logits.float()).all()
