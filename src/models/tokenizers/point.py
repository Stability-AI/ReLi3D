from dataclasses import dataclass

import torch

from src.constants import Names, OutputsType
from src.models.transformers.transformer_1d import Transformer1D
from src.utils.typing import Optional

from .abstract_tokenizer import AbstractTokenizer


class TransformerPointTokenizer(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 64
        in_channels: Optional[int] = 6
        out_channels: Optional[int] = 1024
        num_layers: int = 16
        dropout: float = 0.0
        norm_num_groups: int = 32
        cross_attention_dim: Optional[int] = None
        attention_bias: bool = False
        activation_fn: str = "geglu"
        num_embeds_ada_norm: Optional[int] = None
        cond_dim_ada_norm_continuous: Optional[int] = None
        only_cross_attention: bool = False
        double_self_attention: bool = False
        upcast_attention: bool = False
        norm_type: str = "layer_norm"
        norm_elementwise_affine: bool = True
        attention_type: str = "default"
        enable_memory_efficient_attention: bool = True
        gradient_checkpointing: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()
        transformer_cfg = dict(self.cfg.copy())
        # remove the non-transformer configs
        transformer_cfg["in_channels"] = (
            self.cfg.num_attention_heads * self.cfg.attention_head_dim
        )
        self.model = Transformer1D(transformer_cfg)
        self.linear_in = torch.nn.Linear(
            self.cfg.in_channels, transformer_cfg["in_channels"]
        )
        self.linear_out = torch.nn.Linear(
            transformer_cfg["in_channels"], self.cfg.out_channels
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        points = outputs[Names.POINT_CLOUD]
        assert points.ndim == 3
        inputs = self.linear_in(points).permute(0, 2, 1)  # B N Ci -> B Ci N
        out = self.model(inputs).permute(0, 2, 1)  # B Ci N -> B N Ci
        out = self.linear_out(out)  # B N Ci -> B N Co
        return {self.cfg.tokenize_key: out}
