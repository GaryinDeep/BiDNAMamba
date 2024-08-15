from dataclasses import dataclass, field


@dataclass
class BertConfig:

    vocab_size: int = 50277
    d_model: int = 768
    n_layer: int = 12
    attn_heads : int= 12
    dropout : int =0.1