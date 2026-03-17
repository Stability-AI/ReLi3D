from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.typing import List


class LinearCameraEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        out_channels: int = 0
        conditions: List[FieldName] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def forward(self, outputs: OutputsType) -> OutputsType:
        cond_tensors = []
        for cond_name in self.cfg.conditions:
            assert cond_name in outputs
            assert cond_name.is_cond

            cond = outputs[cond_name]
            # cond in shape (B, Nv, ...)
            cond_tensors.append(cond.view(*cond.shape[:2], -1))
        cond_tensor = torch.cat(cond_tensors, dim=-1)
        assert cond_tensor.shape[-1] == self.cfg.in_channels
        embedding = self.linear(cond_tensor)
        return {Names.CAMERA_EMBEDDING: embedding}

    def consumed_keys(self):
        return super().consumed_keys().union(self.cfg.conditions)

    def produced_keys(self):
        return super().produced_keys().union({Names.CAMERA_EMBEDDING})
