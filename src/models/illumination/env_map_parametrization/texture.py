import abc
import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.utils.typing import List, Union


class TextureAccess(abc.ABC):
    @abc.abstractmethod
    def sample(
        self,
        values: Union[List[Float[Tensor, "B C *D"]], Float[Tensor, "B C *D"]],
        positions: Float[Tensor, "B *D E"],
        **kwargs,
    ) -> Float[Tensor, "B C *D"]:
        pass


class BilinearTextureAccess(TextureAccess):
    def sample(
        self,
        values: Float[Tensor, "B C *D"],
        positions: Float[Tensor, "B *D E"],
        **kwargs,
    ) -> Float[Tensor, "B C *D"]:
        return F.grid_sample(
            values, positions * 2 - 1, align_corners=True, padding_mode="border"
        )


class TrilinearTextureAccess(TextureAccess):
    def sample(
        self,
        values: List[Float[Tensor, "B C *D"]],
        positions: Float[Tensor, "B *D E"],
        level: Union[float, Float[Tensor, "B *D 1"]],
        **kwargs,
    ) -> Float[Tensor, "B C *D"]:
        total_levels = len(values)

        view_shape = [1, 1, 1, -1, *[1 for _ in range(len(positions.shape) - 3)]]
        if isinstance(level, float):
            low_index = max(
                min(int(math.floor(level * total_levels)), total_levels - 1), 0
            )
            upper_index = max(min(low_index + 1, total_levels - 1), 0)
            level_samples: Float[Tensor, "B L C *D"] = torch.stack(
                [
                    F.grid_sample(
                        v, positions * 2 - 1, align_corners=True, padding_mode="border"
                    )
                    for v in values[low_index:upper_index]
                ],
                1,
            )
            low = level_samples[:, 0:1]
            upper = level_samples[:, 1:2]
        else:
            low_index = (level * total_levels).floor().long().clip(0, total_levels - 1)
            upper_index = (low_index + 1).clip(0, total_levels - 1)
            level_samples: Float[Tensor, "B L C *D"] = torch.stack(
                [
                    F.grid_sample(
                        v, positions * 2 - 1, align_corners=True, padding_mode="border"
                    )
                    for v in values
                ],
                1,
            )

            expand_shape = [
                level_samples.shape[0],
                -1,
                level_samples.shape[2],
                -1,
                *[-1 for _ in range(len(positions.shape) - 3)],
            ]
            low = level_samples.gather(
                1,
                low_index.view(*view_shape).expand(expand_shape),
            )
            upper = level_samples.gather(
                1,
                upper_index.view(*view_shape).expand(expand_shape),
            )

        alpha = (level * total_levels - low_index).view(*view_shape)

        ret = (low * (1 - alpha) + upper * alpha).squeeze(1)
        return ret
