import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.ops import normalize
from src.utils.typing import List, Optional, Union

from .abstract_env_map import EnvRepresentationTexture, MipMapEnvRepresentationTexture
from .texture import BilinearTextureAccess, TrilinearTextureAccess


def sign_not_zero(x: Float[Tensor, "*B"]) -> Float[Tensor, "*B"]:  # noqa: F821
    sgn = torch.sign(x)
    # Now make 0 to 1
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    return sgn


def _direction_from_coordinate(
    coordinate: Float[Tensor, "*B 2"],
) -> Float[Tensor, "*B 3"]:
    # OpenGL Convention
    # +X Right
    # +Y Up
    # +Z Backward

    co = coordinate * 2 - 1

    x, y = co.unbind(-1)

    o = torch.stack([x, y, 1 - x.abs() - y.abs()], -1)
    o_alt = torch.stack(
        [
            (1 - y.abs()) * sign_not_zero(x),
            (1 - x.abs()) * sign_not_zero(y),
            o[..., -1],
        ],
        -1,
    )

    return normalize(torch.where(o[..., -1:] < 0, o_alt, o))


def _coordinate_from_direction(
    directions: Float[Tensor, "*B 3"],
) -> Float[Tensor, "*B 2"]:
    # OpenGL Convention
    # +X Right
    # +Y Up
    # +Z Backward

    x, y, z = directions.unbind(-1)

    m = 1 / (x.abs() + y.abs() + z.abs())
    p = torch.stack([x * m, y * m], -1)
    p_x, p_y = p.unbind(-1)
    p_alt = torch.stack(
        [
            (1 - p_y.abs()) * sign_not_zero(p_x),
            (1 - p_x.abs()) * sign_not_zero(p_y),
        ],
        -1,
    )
    return torch.where(z.unsqueeze(-1) <= 0, p_alt, p) * 0.5 + 0.5


class OctahedralEnvRepresentationTexture(EnvRepresentationTexture):
    sampler = BilinearTextureAccess()

    def coordinate_from_direction(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B D"]:
        return _coordinate_from_direction(directions)

    def direction_from_coordinate(
        self, coordinate: Float[Tensor, "*B 2"]
    ) -> Float[Tensor, "*B 3"]:
        return _direction_from_coordinate(coordinate)

    def sample_from_direction(
        self, value: Float[Tensor, "B C R R"], directions: Float[Tensor, "B *N 3"]
    ) -> Float[Tensor, "B C *N"]:
        return self.sample_from_coordinate(
            value, self.coordinate_from_direction(directions)
        )

    def sample_from_coordinate(
        self, value: Float[Tensor, "B C R R"], coordinates: Float[Tensor, "B *N 2"]
    ) -> Float[Tensor, "B C *N"]:
        batch = value.shape[0]
        coordinates_shaped = coordinates.reshape(batch, -1, 1, 2)
        return self.sampler.sample(value, coordinates_shaped).reshape(
            batch,
            value.shape[1],
            *coordinates.shape[1:-1],
        )

    def get_sample_coordinates(
        self, resolution: List[int], device: Optional[torch.device] = None
    ) -> Float[Tensor, "H W 2"]:
        if resolution[1] != resolution[0]:
            raise ValueError("Only square textures are supported")
        return torch.stack(
            torch.meshgrid(
                (torch.arange(resolution[1], device=device) + 0.5) / resolution[1],
                (torch.arange(resolution[0], device=device) + 0.5) / resolution[0],
                indexing="xy",
            ),
            -1,
        )

    def get_resolution(self, value: Float[Tensor, "B C R R"]) -> List[int]:
        return [value.shape[-2], value.shape[-1]]

    def get_distortion_factor(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B 1"]:
        return 4 * torch.pi

    def get_texture_pixel_area(self, value: Float[Tensor, "B C R R"]):
        return value.shape[-2] * value.shape[-1]


class MipMapOctahedralEnvRepresentationTexture(
    MipMapEnvRepresentationTexture, OctahedralEnvRepresentationTexture
):
    sampler = TrilinearTextureAccess()

    def sample_from_direction(
        self,
        value: List[Float[Tensor, "B C R R"]],
        directions: Float[Tensor, "B *N 3"],
        levels: Union[float, Float[Tensor, "B *N 1"]],
    ) -> Float[Tensor, "B C *N"]:
        coordinates = self.coordinate_from_direction(directions)
        return self.sample_from_coordinate(value, coordinates, levels)

    def sample_from_coordinate(
        self,
        value: List[Float[Tensor, "B C R R"]],
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
