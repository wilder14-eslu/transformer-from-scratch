from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Hyperparameters for the encoder-decoder Transformer.

    Defaults match the "base" model from Vaswani et al. (2017).
    """

    d_model: int = 512
    d_head: int = 64
    d_ff: int = 2048
    num_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    enc_max_seq_len: int = 512
    dec_max_seq_len: int = 512
    enc_vocab_dim: int = 10000
    dec_vocab_dim: int = 10000
    pad_id: int = 0
