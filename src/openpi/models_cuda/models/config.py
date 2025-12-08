from dataclasses import dataclass
from dataclasses import field

import torch


@dataclass
class GemmaTextConfig:
    adarms_cond_dim: int | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 2
    eos_token_id: int = 1
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 16384
    max_position_embeddings: int = 8192
    model_type: str = "gemma"
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    num_image_tokens: int = 256
    num_key_value_heads: int = 1
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    torch_dtype: str = "float32"
    use_adarms: bool = False
    use_cache: bool = True
    vocab_size: int = 257152
    ### Added ones
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_fused_geglu: bool = True
    use_cuda_rmsnorm: bool = True


@dataclass
class GemmaExpertConfig:
    adarms_cond_dim: int | None = 1024
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 2
    eos_token_id: int = 1
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 4096
    max_position_embeddings: int = 8192
    model_type: str = "gemma"
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    num_key_value_heads: int = 1
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    torch_dtype: str = "float32"
    transformers_version: str = "4.53.2"
    use_adarms: bool = True
    use_cache: bool = True
    vocab_size: int = 257152
    ### Added ones
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_fused_geglu: bool = True
    use_cuda_rmsnorm: bool = True


@dataclass
class SiglipVisionConfig:
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 224
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    projection_dim: int = 2048
    projector_hidden_act: str = "gelu_fast"
    torch_dtype: torch.dtype = torch.float32
    transformers_version: str = "4.53.2"
    vision_use_head: bool = False
    vocab_size: int = 257152
    ### Added ones
    use_return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_fused_bias_gelu: bool = True
    use_fused_add_layernorm: bool = True


@dataclass
class PaliGemmaConfig:
    _vocab_size: int = 257152
    hidden_size: int = 2048
    image_token_index: int = 257152
    model_type: str = "paligemma"
    projection_dim: int = 2048
    text_config: GemmaTextConfig = field(default_factory=GemmaTextConfig)
    transformers_version: str = "4.53.2"
    vision_config: dict = field(default_factory=SiglipVisionConfig)
