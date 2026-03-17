import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.constants import Names, OutputsType

from .abstract_tokenizer import AbstractTokenizer


class DualSphereLearnablePositionalEmbedding(AbstractTokenizer):
    @dataclass
    class Config(AbstractTokenizer.Config):
        sphere_height: int = 32
        num_channels: int = 1024
        num_planes: int = 1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.embeddings = nn.Parameter(
            torch.randn(
                (
                    2,
                    self.cfg.num_planes,
                    self.cfg.num_channels,
                    self.cfg.sphere_height,
                    self.cfg.sphere_height * 2,
                ),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        batch_size = outputs[Names.BATCH_SIZE]
        return {
            self.cfg.tokenize_key: rearrange(
                repeat(
                    self.embeddings, "Np Dp Ct Hp Wp -> B Np Dp Ct Hp Wp", B=batch_size
                ),
                "B Np Dp Ct Hp Wp -> B Ct (Dp Np Hp Wp)",
            )
        }

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        tokens = outputs[self.cfg.detokenize_key]
        batch_size, Ct, Nt = tokens.shape
        assert (
            Nt
            == (
                self.cfg.num_planes
                * self.cfg.sphere_height
                * self.cfg.sphere_height
                * 2
            )
            * 2
        )
        assert Ct == self.cfg.num_channels
        return {
            self.cfg.detokenize_key: rearrange(
                tokens,
                "B Ct (Dp Np Hp Wp) -> B Np Dp Ct Hp Wp",
                Np=2,
                Dp=self.cfg.num_planes,
                Hp=self.cfg.sphere_height,
                Wp=self.cfg.sphere_height * 2,
            )
        }
