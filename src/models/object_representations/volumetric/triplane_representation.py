from dataclasses import dataclass

import torch
from einops import rearrange, reduce
from jaxtyping import Float
from torch import Tensor
from torch.functional import F

from src.constants import FieldName, Names, OutputsType
from src.utils.ops import scale_tensor
from src.utils.shape_utils import check_shape
from src.utils.typing import List, Literal

from .abstract_volumetric_representation import AbstractNeuralVolumetricRepresentation


def query_triplane(
    positions: Float[Tensor, "*B N 3"],
    triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    radius: float,
    feature_reduction: Literal["concat", "mean", "sum"],
) -> Float[Tensor, "*B N F"]:
    batched_positions = positions.ndim == 3
    batched_triplanes = triplanes.ndim == 5
    if not batched_positions:
        # no batch dimension
        positions = positions[None, ...]
    if not batched_triplanes:
        # no batch dimension
        triplanes = triplanes[None, ...]
    assert (
        triplanes.ndim == 5 and positions.ndim == 3
    ), f"triplanes: {triplanes.shape}, positions: {positions.shape}"

    positions = scale_tensor(positions, (-radius, radius), (-1, 1))

    # For each point, we sample from 3 orthogonal planes:
    # 1. XY plane: Using x,y coordinates to sample features
    # 2. XZ plane: Using x,z coordinates to sample features
    # 3. YZ plane: Using y,z coordinates to sample features
    indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
        (
            positions[..., [0, 1]],
            positions[..., [0, 2]],
            positions[..., [1, 2]],
        ),
        dim=-3,
    ).to(triplanes.dtype)
    out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
        rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).float(),
        rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3).float(),
        align_corners=False,
        mode="bilinear",
    )
    if feature_reduction == "concat":
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
    elif feature_reduction == "mean":
        out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
    elif feature_reduction == "sum":
        out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="sum")
    else:
        raise NotImplementedError

    if not batched_positions:
        out = out.squeeze(0)
    return out


class VolumetricTriplaneRepresentation(AbstractNeuralVolumetricRepresentation):
    @dataclass
    class Config(AbstractNeuralVolumetricRepresentation.Config):
        triplane_features: int = 128
        feature_reduction: str = "concat"  # "concat", "mean", "sum"

    cfg: Config

    def configure(self):
        super().configure(
            feature_dimension=self.cfg.triplane_features * 3
            if self.cfg.feature_reduction == "concat"
            else self.cfg.triplane_features
        )

    def consumed_keys(self):
        return super().consumed_keys().union({Names.TRIPLANE})

    def forward_impl(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
        include: List[FieldName] = None,
        exclude: List[FieldName] = None,
    ) -> OutputsType:
        triplanes = outputs[Names.TRIPLANE]
        shape_test = check_shape(outputs, Names.TRIPLANE, known_dims=4)
        assert (
            shape_test.has_known_dims
            and not shape_test.has_other_dims
            and not shape_test.has_view_dim
        ), f"Triplanes have unknown dimensions, {triplanes.shape}"

        triplane_values = query_triplane(
            positions, triplanes, self.cfg.radius, self.cfg.feature_reduction
        )
        net_values = self.net(
            {Names.TOKEN: triplane_values},
            include=include,
            exclude=exclude,
        )

        return net_values
