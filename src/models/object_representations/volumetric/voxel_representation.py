from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
from torch.functional import F

from src.constants import FieldName, Names, OutputsType
from src.utils.ops import scale_tensor
from src.utils.shape_utils import check_shape
from src.utils.typing import List

from .abstract_volumetric_representation import AbstractNeuralVolumetricRepresentation


def query_voxel_grid(
    positions: Float[Tensor, "*B N 3"],
    voxel_grid: Float[Tensor, "*B C H W D"],
    radius: float,
) -> Float[Tensor, "*B N C"]:
    batched_positions = positions.ndim == 3
    batched_voxel_grid = voxel_grid.ndim == 5
    if not batched_positions:
        # no batch dimension
        positions = positions[None, ...]
    if not batched_voxel_grid:
        # no batch dimension
        voxel_grid = voxel_grid[None, ...]
    assert (
        voxel_grid.ndim == 5 and positions.ndim == 3
    ), f"voxel_grid: {voxel_grid.shape}, positions: {positions.shape}"

    # Scale positions from world space to grid space [-1, 1]
    positions = scale_tensor(positions, (-radius, radius), (-1, 1))

    # Grid sample expects 5D input [B, C, D, H, W]
    # Add two singleton dimensions for height and width
    positions = positions.unsqueeze(1).unsqueeze(1)

    # Sample from the 3D grid using trilinear interpolation
    samples: Float[Tensor, "B C 1 1 N"] = F.grid_sample(
        voxel_grid.float(),
        positions.float(),
        align_corners=False,
        mode="bilinear",
    )

    # Remove singleton dimensions and transpose to match expected output
    out = samples.squeeze(2).squeeze(2).transpose(-1, -2)

    if not batched_positions:
        out = out.squeeze(0)

    return out


class VolumetricVoxelRepresentation(AbstractNeuralVolumetricRepresentation):
    @dataclass
    class Config(AbstractNeuralVolumetricRepresentation.Config):
        voxel_features: int = 128

    cfg: Config

    def configure(self):
        super().configure(feature_dimension=self.cfg.voxel_features)

    def consumed_keys(self):
        return super().consumed_keys().union({Names.VOXEL_GRID})

    def forward_impl(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
        include: List[FieldName] = None,
        exclude: List[FieldName] = None,
    ) -> OutputsType:
        voxel_grid = outputs[Names.VOXEL_GRID]
        shape_test = check_shape(outputs, Names.VOXEL_GRID, known_dims=4)
        assert (
            shape_test.has_known_dims
            and not shape_test.has_other_dims
            and not shape_test.has_view_dim
        ), f"Voxel grid has unknown dimensions, {voxel_grid.shape}"

        voxel_values = query_voxel_grid(positions, voxel_grid, self.cfg.radius)
        net_values = self.net(
            {Names.TOKEN: voxel_values},
            include=include,
            exclude=exclude,
        )

        return net_values
