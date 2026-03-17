from dataclasses import dataclass, field

import torch

from src.constants import Names, OutputsType
from src.models.background.base import BaseBackground
from src.utils.typing import List


class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: List[float] = field(default_factory=[1.0, 1.0, 1.0])

    cfg: Config

    def configure(self) -> None:
        self.register_buffer(
            "env_color",
            torch.as_tensor(self.cfg.color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        outputs: OutputsType,
    ) -> OutputsType:
        size = []
        if Names.BATCH_SIZE in outputs:
            size.append(outputs[Names.BATCH_SIZE])
        if Names.VIEW_SIZE in outputs:
            size.append(outputs[Names.VIEW_SIZE])
        size += [
            outputs[Names.HEIGHT],
            outputs[Names.WIDTH],
            self.cfg.n_output_dims,
        ]
        color = self.env_color.view(
            *[1] * (len(size) - 1) + [self.cfg.n_output_dims]
        ).expand(size)
        return {Names.BACKGROUND: color}
