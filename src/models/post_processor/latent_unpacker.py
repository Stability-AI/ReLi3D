from dataclasses import dataclass, field

import numpy as np

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.ops import get_activation
from src.utils.typing import List


class LatentUnpacker(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        keys: List[FieldName] = field(default_factory=list)
        unpack_key: FieldName = Names.LATENTS
        unpack_shape: List[int] = field(default_factory=list)
        shapes: List[str] = field(default_factory=list)
        activation: List[str] = field(default_factory=list)
        out_bias: List[float] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.shapes = [
            [int(s) for s in shape.split(",") if s != ""] for shape in self.cfg.shapes
        ]
        if not self.cfg.activation:
            self.cfg.activation = ["none"] * len(self.cfg.keys)
        self.activation = [get_activation(a) for a in self.cfg.activation]
        if not self.cfg.out_bias:
            self.cfg.out_bias = [0.0] * len(self.cfg.keys)
        self.out_bias = [float(b) for b in self.cfg.out_bias]

    def forward(self, outputs: OutputsType) -> OutputsType:
        ret = {}
        start = 0
        batch_size = outputs[Names.BATCH_SIZE]
        for k, shape, activation, bias in zip(
            self.cfg.keys, self.shapes, self.activation, self.out_bias
        ):
            num_elements = np.prod(shape)
            ret[k] = activation(
                outputs[self.cfg.unpack_key]
                .view(batch_size, *self.cfg.unpack_shape)[
                    ..., start : start + num_elements
                ]
                .view(batch_size, *shape)
                + bias
            )
            start += num_elements
        return ret

    def produced_keys(self):
        return super().produced_keys().union(self.cfg.keys)

    def consumed_keys(self):
        return super().consumed_keys().union({self.cfg.unpack_key})
