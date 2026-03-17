import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.constants import Names, OutputsType
from src.utils.base import BaseModule
from src.utils.typing import Dict

from ..env_map_parametrization.octahedral import OctahedralEnvRepresentationTexture
from ..env_map_parametrization.spherical import SphericalEnvRepresentationTexture
from .field import RENIField

ENV_PARAMETRIZATION = {
    "spherical": SphericalEnvRepresentationTexture,
    "octahedral": OctahedralEnvRepresentationTexture,
}


class RENIEnvMap(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        reni_config: dict = field(default_factory=dict)
        parametrization: str = "spherical"
        """Which env parametrization to use: spherical, cubemap, octahedral"""
        resolution: int = 128

    cfg: Config

    def configure(self):
        super().configure()
        self.field = RENIField(RENIField.Config(**self.cfg.reni_config))
        self.env_parametrization = ENV_PARAMETRIZATION[self.cfg.parametrization]()
        resolution = (
            (self.cfg.resolution, self.cfg.resolution)
            if self.cfg.parametrization != "spherical"
            else (self.cfg.resolution, self.cfg.resolution * 2)
        )
        sample_directions = self.env_parametrization.direction_from_coordinate(
            self.env_parametrization.get_sample_coordinates(resolution)
        )
        self.img_shape = sample_directions.shape[:-1]

        sample_directions_flat = sample_directions.view(-1, 3)

        # FIXME: (Jan) March 31, 2025
        # This is a hot fix to ensure the rotation is applied correctly.
        # It is 270 degrees around the y-axis

        angle = 270
        theta = math.radians(angle)

        rotation_matrix = torch.tensor(
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ],
            dtype=sample_directions_flat.dtype,
            device=sample_directions_flat.device,
        )

        sample_directions_flat = sample_directions_flat @ rotation_matrix

        # Lastly these have y up but reni expects z up. Rotate 90 degrees on x axis
        sample_directions_flat = torch.stack(
            [
                sample_directions_flat[:, 0],
                -sample_directions_flat[:, 2],
                sample_directions_flat[:, 1],
            ],
            -1,
        )

        self.sample_directions = torch.nn.Parameter(
            sample_directions_flat, requires_grad=False
        )

    def forward(self, outputs: OutputsType) -> Dict[str, Tensor]:
        # FIXME: (Jan) March 31, 2025
        # This is a hot fix to ensure the rotation is applied correctly.
        # It rotates the illumination in the inverse direction of the z-axis

        if Names.ILLUMINATION_Z_ROTATION_RADS not in outputs:
            theta_z = torch.zeros((outputs[Names.BATCH_SIZE],)).to(self.device)
        else:
            illumination_z_rotation_rads = outputs[Names.ILLUMINATION_Z_ROTATION_RADS]
            theta_z = -illumination_z_rotation_rads * 2

        # Compute cosine, sine and constant tensors for the batch (theta_z shape: [B])
        cos_t = torch.cos(theta_z)  # shape: [B]
        sin_t = torch.sin(theta_z)  # shape: [B]
        zeros = torch.zeros_like(theta_z)
        ones = torch.ones_like(theta_z)

        # Build a batch of rotation matrices of shape [B, 3, 3]
        rotation_matrix = torch.stack(
            [
                torch.stack([cos_t, -sin_t, zeros], dim=-1),
                torch.stack([sin_t, cos_t, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        )  # shape: [B, 3, 3]

        # Assume self.sample_directions has shape [N, 3]. Expand to [B, N, 3]
        B = outputs[Names.BATCH_SIZE]
        sample_directions_expanded = self.sample_directions.unsqueeze(0).expand(
            B, -1, -1
        )

        # Apply the batch of rotation matrices:
        # For each batch element, multiply the sample_directions (row vectors) by its rotation matrix.
        sample_directions_rotated = torch.bmm(
            sample_directions_expanded, rotation_matrix
        )  # shape: [B, N, 3]

        return {
            k: v.view(outputs[Names.BATCH_SIZE], *self.img_shape, -1)
            for k, v in self.field(
                outputs | {Names.DIRECTION.add_suffix("env"): sample_directions_rotated}
            ).items()
        }

    def consumed_keys(self):
        return super().consumed_keys().union(self.field.consumed_keys()) | {
            Names.ILLUMINATION_Z_ROTATION_RADS
        }

    def produced_keys(self):
        return (
            super()
            .produced_keys()
            .union(
                self.field.produced_keys().union({Names.DIRECTION.add_suffix("env")})
            )
        )
