import typing

import torch
from jaxtyping import Float
from torch import Tensor

from .ops import coordinate_system, dot, safe_sqrt


class Frame:
    def __init__(
        self,
        normal: Float[Tensor, "*B 3"],
        s: typing.Optional[Float[Tensor, "*B 3"]] = None,
        t: typing.Optional[Float[Tensor, "*B 3"]] = None,
    ):
        super().__init__()

        self._normal = normal
        if s is None or t is None:
            self._s, self._t = coordinate_system(normal)
        else:
            self._s = s
            self._t = t

    def to_local(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        assert self._normal.dim() == v.dim(), (
            f"The coordinate frame was created with a rank of {self._normal.dim()} "
            f"and is now queried with {v.dim()}. The ranks should match."
        )
        return torch.cat((dot(v, self._s), dot(v, self._t), dot(v, self._normal)), -1)

    def to_world(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        assert self._normal.dim() == v.dim(), (
            f"The coordinate frame was created with a rank of {self._normal.dim()} "
            f"and is now queried with {v.dim()}. The ranks should match."
        )
        vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
        return self._s * vx + self._t * vy + self._normal * vz

    def cos_theta(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the cosine
            of the angle between normal and v is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Cosine of the angle between normal and v
        """
        return v[..., 2:3]

    def sin_theta(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the sine
            of the angle between normal and v is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Sine of the angle between normal and v
        """
        temp = self.sin_theta2(v)
        temp = torch.where(temp <= 0, torch.zeros_like(temp), temp)
        return temp.sqrt()

    def tan_theta(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the tangent
            of the angle between normal and v is returned

        Args:
            v (TensorType["B": ..., 3]): A unit vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Tangent of the angle between normal and v
        """
        vz = v[..., 2:3]

        temp = 1 - vz * vz
        return safe_sqrt(temp) / vz

    def sin_theta2(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the squared
            sine of the angle between normal and v is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Squared sine of the angle between normal and v
        """
        vz = v[..., 2:3]
        return 1.0 - vz * vz

    def sin_phi(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the sine
           of the phi parameter in spherical coordinates is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Sine of the phi parameter
        """
        sin_theta = self.sin_theta(v)

        vy = v[..., 1:2]
        return torch.where(
            sin_theta == 0, torch.ones_like(sin_theta), (vy / sin_theta).clip(-1, 1)
        )

    def cos_phi(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the
            cosine of the phi parameter in spherical coordinates is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Cosine of the phi parameter
        """
        sin_theta = self.sin_theta(v)

        vx = v[..., 0:1]
        return torch.where(
            sin_theta == 0, torch.ones_like(sin_theta), (vx / sin_theta).clip(-1, 1)
        )

    def sin_phi2(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the squared
           sine of the phi parameter in spherical coordinates is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Squared sine of the phi parameter
        """
        vy = v[..., 1:2]
        return (vy * vy / self.sin_theta2(v)).clip(0, 1)

    def cos_phi2(self, v: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 1"]:
        """Assume that v is in the local coordinate system, then the squared
           cosine of the phi parameter in spherical coordinates is returned

        Args:
            v (TensorType["B": ..., 3]): A vector in the local coordinate system

        Returns:
            TensorType["B": ..., 1]: Squared cosine of the phi parameter
        """
        vx = v[..., 0:1]
        return (vx * vx / self.sin_theta2(v)).clip(0, 1)
