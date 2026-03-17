import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from src.constants import FieldName, Names, OutputsType
from src.utils.ops import scale_tensor
from src.utils.shape_utils import check_shape
from src.utils.typing import List

from .abstract_volumetric_representation import AbstractNeuralVolumetricRepresentation


def rotx_np(alpha: float) -> np.ndarray:
    """Generate 3D rotation matrix around X axis."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def roty_np(beta: float) -> np.ndarray:
    """Generate 3D rotation matrix around Y axis."""
    return np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )


def rotz_np(gamma: float) -> np.ndarray:
    """Generate 3D rotation matrix around Z axis."""
    return np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )


def generate_planes() -> torch.Tensor:
    """
    Define planes by the three vectors that form the "axes" of the plane.
    Works with arbitrary number of planes and planes of arbitrary orientation.
    """
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )


def project_onto_planes(
    planes: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """
    Project 3D points onto a batch of 2D planes.

    Args:
        planes: Plane axes of shape (n_planes, 3, 3)
        coordinates: Points of shape (N, M, 3)

    Returns:
        Projections of shape (N*n_planes, M, 2)
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = (
        coordinates.unsqueeze(1)
        .expand(-1, n_planes, -1, -1)
        .reshape(N * n_planes, M, 3)
    )
    inv_planes = (
        torch.linalg.inv(planes)
        .unsqueeze(0)
        .expand(N, -1, -1, -1)
        .reshape(N * n_planes, 3, 3)
    )
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def cartesian_to_spherical(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Cartesian coordinates to spherical coordinates."""
    # radius [0., sqrt(3)] -> [-1, 1]
    radius = (coordinates**2).sum(axis=-1).sqrt()
    radius = radius / math.sqrt(3)
    radius = 2.0 * radius - 1.0

    # theta: [0, pi] -> [-1, 1]
    theta = torch.atan2(
        (coordinates[:, :, :2] ** 2).sum(dim=-1).sqrt(), coordinates[:, :, 2]
    )
    theta = theta / math.pi
    theta = 2.0 * theta - 1.0

    # phi: (-pi, pi] -> [-1, 1]
    phi = torch.atan2(coordinates[:, :, 1], coordinates[:, :, 0])
    phi = phi / math.pi

    return theta, phi, radius


def spherical_weights(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Calculate blending weights for spherical coordinates."""
    tw = (1.0 + torch.cos(torch.pi * theta)) / 2.0
    pw = (1.0 + torch.cos(torch.pi * phi)) / 2.0
    return tw * pw


def sample_from_spheres(
    plane_features: torch.Tensor,
    coordinates: torch.Tensor,
    plane_axes: torch.Tensor,
    mode: str,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample features from spherical representation."""
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    plane_features = plane_features.reshape(N * n_planes, C, H, W)

    theta, phi, radius = cartesian_to_spherical(coordinates)
    w = spherical_weights(theta, phi).view(N, 1, M, 1).expand(-1, n_planes, -1, C)
    projected_coordinates = torch.stack([theta, phi, radius], dim=-1)
    projected_coordinates = project_onto_planes(
        plane_axes, projected_coordinates
    ).unsqueeze(1)

    output_features = (
        torch.nn.functional.grid_sample(
            plane_features,
            projected_coordinates.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        .permute(0, 3, 2, 1)
        .reshape(N, n_planes, M, C)
    )

    return output_features, w


def sample_from_dual_spheres(
    plane_features: torch.Tensor,
    coordinates: torch.Tensor,
    plane_axes: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    box_warp: float = None,
) -> torch.Tensor:
    """Sample features from dual sphere representation."""
    assert padding_mode == "zeros"
    N, n_planes, D, C, H, W = plane_features.shape
    assert n_planes == 2
    _, M, _ = coordinates.shape

    plane_features_A = plane_features[:, 0, :, :, :].view(N, D, C, H, W)
    plane_features_B = plane_features[:, 1, :, :, :].view(N, D, C, H, W)

    rmat_A = torch.from_numpy((roty_np(np.pi / 2.0)).T.astype(np.float32)).to(
        coordinates.device
    )
    rmat_B = torch.from_numpy(
        (roty_np(-np.pi / 2) @ rotz_np(-np.pi / 2.0)).T.astype(np.float32)
    ).to(coordinates.device)
    coordinates = (2 / box_warp) * coordinates

    coordinates_A = coordinates @ rmat_A
    coordinates_B = coordinates @ rmat_B

    output_features_A, w_A = sample_from_spheres(
        plane_features_A, coordinates_A, plane_axes, mode, padding_mode
    )
    output_features_B, w_B = sample_from_spheres(
        plane_features_B, coordinates_B, plane_axes, mode, padding_mode
    )

    # Blend features using weights
    fused_features = (output_features_A * w_A) + (output_features_B * w_B)
    fused_features = fused_features / (w_A + w_B + 1.0e-8)

    return fused_features


def query_dual_sphere(
    positions: Float[Tensor, "*B N 3"],
    sphere_features: Float[Tensor, "*B 2 D C H W"],
    plane_axes: Float[Tensor, "3 3"],
    radius: float,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Float[Tensor, "*B N F"]:
    """Query features from dual sphere representation at given positions."""
    batched_positions = positions.ndim == 3
    batched_sphere_features = sphere_features.ndim == 5
    if not batched_positions:
        # no batch dimension
        positions = positions[None, ...]
    if not batched_sphere_features:
        # no batch dimension
        sphere_features = sphere_features[None, ...]
    assert (
        sphere_features.ndim == 5 and positions.ndim == 3
    ), f"sphere_features: {sphere_features.shape}, positions: {positions.shape}"

    positions = scale_tensor(positions, (-radius, radius), (-1, 1))

    output_features = sample_from_dual_spheres(
        sphere_features,
        positions,
        plane_axes,
        mode=mode,
        padding_mode=padding_mode,
        box_warp=2.0,  # Since we scaled to [-1, 1]
        triplane_depth=1,
    )

    if not batched_positions:
        output_features = output_features.squeeze(0)

    return output_features


class VolumetricDualSphereRepresentation(AbstractNeuralVolumetricRepresentation):
    @dataclass
    class Config(AbstractNeuralVolumetricRepresentation.Config):
        sphere_features: int = 128
        sampling_mode: str = "bilinear"
        padding_mode: str = "zeros"

    cfg: Config

    def configure(self):
        super().configure(feature_dimension=self.cfg.sphere_features)
        self.register_buffer("plane_axes", generate_planes())

    def consumed_keys(self):
        return super().consumed_keys().union({Names.DUAL_SPHERE})

    def forward_impl(
        self,
        outputs: OutputsType,
        positions: Float[Tensor, "*B N 3"],
        include: List[FieldName] = None,
        exclude: List[FieldName] = None,
    ) -> OutputsType:
        shape_test = check_shape(outputs, Names.DUAL_SPHERE, known_dims=5)
        assert (
            shape_test.has_known_dims
            and not shape_test.has_other_dims
            and not shape_test.has_view_dim
        ), f"Dual sphere has unknown dimensions, {outputs[Names.DUAL_SPHERE].shape}"

        sphere_features = outputs[Names.DUAL_SPHERE]
        sphere_values = query_dual_sphere(
            positions,
            sphere_features,
            self.plane_axes,
            self.cfg.radius,
            self.cfg.sampling_mode,
            self.cfg.padding_mode,
        )
        return self.net(
            {Names.TOKEN: sphere_values},
            include=include,
            exclude=exclude,
        )
