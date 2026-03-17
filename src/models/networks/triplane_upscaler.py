from dataclasses import dataclass

import torch.nn as nn
from einops import rearrange

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.typing import Float, Tensor


class SimpleConvUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 80
        upsample: bool = True
        conv_kernel_size: int = 1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if self.cfg.upsample:
            self.upsample = nn.ConvTranspose2d(
                self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
            )
        else:
            self.upsample = nn.Conv2d(
                self.cfg.in_channels,
                self.cfg.out_channels,
                kernel_size=self.cfg.conv_kernel_size,
                stride=1,
                padding=self.cfg.conv_kernel_size // 2,
            )

    def forward(
        self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]
    ) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
        return rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )


class PixelShuffleUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 80
        scale_factor: int = 2

        conv_layers: int = 1
        conv_kernel_size: int = 3
        triplane_key: FieldName = Names.TRIPLANE

    cfg: Config

    def configure(self) -> None:
        super().configure()
        layers = []
        output_channels = self.cfg.out_channels * self.cfg.scale_factor**2

        in_channels = self.cfg.in_channels
        for i in range(self.cfg.conv_layers):
            cur_out_channels = (
                in_channels if i != self.cfg.conv_layers - 1 else output_channels
            )
            layers.append(
                nn.Conv2d(
                    in_channels,
                    cur_out_channels,
                    self.cfg.conv_kernel_size,
                    padding=(self.cfg.conv_kernel_size - 1) // 2,
                )
            )
            if i != self.cfg.conv_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.PixelShuffle(self.cfg.scale_factor))

        self.upsample = nn.Sequential(*layers)

    def forward(self, outputs: OutputsType) -> OutputsType:
        triplanes = outputs[self.cfg.triplane_key]
        return {
            self.cfg.triplane_key.raw: triplanes,
            self.cfg.triplane_key: rearrange(
                self.upsample(
                    rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
                ),
                "(B Np) Co Hp Wp -> B Np Co Hp Wp",
                Np=3,
            ),
        }

    def consumed_keys(self):
        return super().consumed_keys() | {self.cfg.triplane_key}

    def produced_keys(self):
        return super().produced_keys() | {
            self.cfg.triplane_key,
            self.cfg.triplane_key.raw,
        }
