from dataclasses import dataclass, field


@dataclass
class hyenadnaConfig:

    d_model: int = 256
    n_layer: int = 8
    d_inner: int = 256*4
    vocab_size: int = 50277
    layer: dict = None
    attn_layer_idx: list = None
    attn_cfg: dict = None
    max_position_embeddings: int = 0
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1 
    layer_norm_epsilon: float = 1e-5 
    residual_in_fp32: bool = False 
    pad_vocab_size_multiple: int = 1 
    tie_embeddings: bool = True
    seq_len_max : int = None
