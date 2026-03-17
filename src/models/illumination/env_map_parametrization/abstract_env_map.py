import abc

import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.typing import List, Optional, Union


class EnvRepresentationTexture(abc.ABC):
    @abc.abstractmethod
    def coordinate_from_direction(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B D"]:
        pass

    @abc.abstractmethod
    def direction_from_coordinate(
        self, coordinate: Float[Tensor, "*B D"]
    ) -> Float[Tensor, "*B 3"]:
        pass

    @abc.abstractmethod
    def get_sample_coordinates(
        self, resolution: List[int], device: Optional[torch.device] = None
    ) -> Float[Tensor, "B *N D"]:
        pass

    @abc.abstractmethod
    def sample_from_direction(
        self, value: Float[Tensor, "B C *D"], directions: Float[Tensor, "B *N 3"]
    ) -> Float[Tensor, "B C *N"]:
        pass

    @abc.abstractmethod
    def sample_from_coordinate(
        self, value: Float[Tensor, "B C *D"], coordinates: Float[Tensor, "B *N D"]
    ) -> Float[Tensor, "B C *N"]:
        pass

    @abc.abstractmethod
    def get_resolution(self, value: Float[Tensor, "B C *D"]) -> List[int]:
        pass

    @abc.abstractmethod
    def get_distortion_factor(
        self, directions: Float[Tensor, "*B 3"]
    ) -> Float[Tensor, "*B 1"]:
        pass

    @abc.abstractmethod
    def get_texture_pixel_area(self, value: Float[Tensor, "B C *D"]) -> int:
        pass


class MipMapEnvRepresentationTexture(EnvRepresentationTexture):
    @abc.abstractmethod
    def sample_from_direction(
        self,
        value: List[Float[Tensor, "B C *D"]],
        directions: Float[Tensor, "B *N 3"],
        levels: Union[float, Float[Tensor, "B *N 1"]],
    ) -> Float[Tensor, "B C *N"]:
        pass

    @abc.abstractmethod
    def sample_from_coordinate(
        self,
        value: List[Float[Tensor, "B C *D"]],
        coordinates: Float[Tensor, "B *N D"],
        levels: Union[float, Float[Tensor, "B *N 1"]],
    ) -> Float[Tensor, "B C *N"]:
        pass
