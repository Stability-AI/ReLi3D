import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.typing import List, Optional, Union

from .abstract_env_map import EnvRepresentationTexture, MipMapEnvRepresentationTexture
from .texture import BilinearTextureAccess, TrilinearTextureAccess


def _direction_from_coordinate(
    coordinate: Float[Tensor, "*B 2"],
) -> Float[Tensor, "*B 3"]:
    # OpenGL Convention
    # +X Right
    # +Y Up
    # +Z Backward

    u, v = coordinate.unbind(-1)
    theta = (2 * torch.pi * u) - torch.pi
    phi = torch.pi * v

    dir = torch.stack(
        [
            theta.sin() * phi.sin(),
            phi.cos(),
            -1 * theta.cos() * phi.sin(),
        ],
        -1,
    )
    return dir


def _coordinate_from_direction(
    directions: Float[Tensor, "*B 3"],
) -> Float[Tensor, "*B 2"]:
    # OpenGL Convention
    # +X Right
    # +Y Up
    # +Z Backward
    x, y, z = directions.unbind(-1)
    theta = torch.atan2(x, z)
    phi = torch.acos(y)

    theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

    x = theta / (2 * torch.pi)
    y = phi / torch.pi

    return torch.stack([1 - x, y], -1)


class SphericalEnvRepresentationTexture(EnvRepresentationTexture):
    sampler = BilinearTextureAccess()

    def coordinate_from_direction(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B D"]:
        return _coordinate_from_direction(directions)

    def direction_from_coordinate(
        self, coordinate: Float[Tensor, "*B D"]
    ) -> Float[Tensor, "*B 3"]:
        return _direction_from_coordinate(coordinate)

    def sample_from_direction(
        self, value: Float[Tensor, "B C H W"], directions: Float[Tensor, "B *N 3"]
    ) -> Float[Tensor, "B C *N"]:
        return self.sample_from_coordinate(
            value, self.coordinate_from_direction(directions)
        )

    def sample_from_coordinate(
        self, value: Float[Tensor, "B C H W"], coordinates: Float[Tensor, "B *N 2"]
    ) -> Float[Tensor, "B C *N"]:
        batch = value.shape[0]
        coordinates_shaped = coordinates.view(batch, -1, 1, 2)
        return self.sampler.sample(value, coordinates_shaped).view(
            batch,
            value.shape[1],
            *coordinates.shape[1:-1],
        )

    def get_sample_coordinates(
        self, resolution: List[int], device: Optional[torch.device] = None
    ) -> Float[Tensor, "H W 2"]:
        return torch.stack(
            torch.meshgrid(
                (torch.arange(resolution[1], device=device) + 0.5) / resolution[1],
                (torch.arange(resolution[0], device=device) + 0.5) / resolution[0],
                indexing="xy",
            ),
            -1,
        )

    def get_resolution(self, value: Float[Tensor, "B C H W"]) -> List[int]:
        return [value.shape[-2], value.shape[-1]]

    def get_distortion_factor(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B 1"]:
        x, y, z = directions.unbind(-1)
        distortion = (1 - y.square()).sqrt() * (
            x.square() + y.square() + z.square()
        ).sqrt()
        return distortion.unsqueeze(-1)

    def get_texture_pixel_area(self, value: Float[Tensor, "B C H W"]):
        return value.shape[-2] * value.shape[-1]


class MipMapSphericalEnvRepresentationTexture(
    MipMapEnvRepresentationTexture, SphericalEnvRepresentationTexture
):
    sampler = TrilinearTextureAccess()

    def sample_from_direction(
        self,
        value: List[Float[Tensor, "B C H W"]],
        directions: Float[Tensor, "B *N 3"],
        levels: Union[float, Float[Tensor, "B *N 1"]],
    ) -> Float[Tensor, "B C *N"]:
        coordinates = self.coordinate_from_direction(directions)
        return self.sample_from_coordinate(value, coordinates, levels)

    def sample_from_coordinate(
        self,
        value: List[Float[Tensor, "B C H W"]],
        coordinates: Float[Tensor, "B *N 2"],
        levels: Union[float, Float[Tensor, "B *N 1"]],
    ) -> Float[Tensor, "B C *N"]:
        batch = value[0].shape[0]
        coordinates_shaped = coordinates.view(batch, -1, 1, 2)
        if isinstance(levels, float):
            levels_shaped = levels
        else:
            levels_shaped = levels.view(batch, -1, 1, 1)
        return self.sampler.sample(
            value,
            coordinates_shaped,
            level=levels_shaped,
        ).view(batch, value[0].shape[1], *coordinates.shape[1:-1])

    def get_texture_pixel_area(self, value: List[Float[Tensor, "B C H W"]]):
        return value[0].shape[-2] * value[0].shape[-1]
