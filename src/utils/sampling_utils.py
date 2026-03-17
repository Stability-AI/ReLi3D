from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.halton_sequence import create_halton_sequence
from src.utils.ops import safe_sqrt
from src.utils.typing import Dict, NamedTuple, Optional, Union


class Samples(NamedTuple):
    sample_direction: Float[Tensor, "*B S 3"]
    pdf: Float[Tensor, "*B S 1"]

    def detach(self) -> Samples:
        return Samples(self.sample_direction.detach(), self.pdf.detach())


class AbstractSampleMutationStrategy(torch.nn.Module, ABC):
    @abstractmethod
    def permutate(
        self, batch_size: int, samples: Float[Tensor, "num_samples dim"]
    ) -> Float[Tensor, "batch num_samples dim"]:
        pass


class CranleyPattersonRotationMutationStrategy(AbstractSampleMutationStrategy):
    def __init__(self, scale: float = 0.025) -> None:
        super().__init__()
        self._scale = scale

    def permutate(
        self, batch_size: int, samples: Float[Tensor, "num_samples dim"]
    ) -> Float[Tensor, "batch num_samples dim"]:
        rand_points = (
            torch.randn(batch_size, *samples.shape, device=samples.device) * self._scale
        )
        offset_samples = samples.unsqueeze(0) + rand_points
        return offset_samples % 1


class AbstractSampler(torch.nn.Module, ABC):
    def __init__(
        self,
        mutation_strategy: Optional[AbstractSampleMutationStrategy] = None,
        cache_samples: bool = True,
    ):
        """Abstract sampler class for generating 2D or 1D random samples

        Args:
            mutation_strategy (AbstractSampleMutationStrategy, optional): Instead of drawing samples
                for each batch, we mutate a base set of samples for each batch. For expensive samplers
                this can speed up training.
                Defaults to None (no mutation).
            cache_samples (bool, optional): Whether to cache samples. Defaults to True.
        """
        super().__init__()
        self._mutation_strategy = mutation_strategy
        self._sample_store: Dict[str, Float[Tensor, "num_samples dim"]] = {}
        self._cache_samples = cache_samples

    def get_samples_1D(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        if self._mutation_strategy is not None:
            if self._cache_samples:
                sample_key = f"1D-{num_samples}"
                if sample_key not in self._sample_store:
                    samples = self.samples_1D_impl(
                        1, num_samples, device=device, dtype=dtype
                    )
                    self._sample_store[sample_key] = samples
                else:
                    samples = self._sample_store[sample_key]
            else:
                samples = self.samples_1D_impl(
                    1, num_samples, device=device, dtype=dtype
                )

            return self._mutation_strategy.permutate(batch_size, samples.squeeze(0))

        return self.samples_1D_impl(batch_size, num_samples, device=device, dtype=dtype)

    @abstractmethod
    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        pass

    def get_samples_2D(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        if self._mutation_strategy is not None:
            if self._cache_samples:
                sample_key = f"2D-{num_samples}"
                if sample_key not in self._sample_store:
                    samples = self.samples_2D_impl(
                        1, num_samples, device=device, dtype=dtype
                    )
                    self._sample_store[sample_key] = samples
                else:
                    samples = self._sample_store[sample_key]
            else:
                samples = self.samples_2D_impl(
                    1, num_samples, device=device, dtype=dtype
                )

            return self._mutation_strategy.permutate(batch_size, samples.squeeze(0))

        return self.samples_2D_impl(batch_size, num_samples, device=device, dtype=dtype)

    @abstractmethod
    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        pass


class SobolSampler(AbstractSampler):
    """Uses the sobol sequence for generating samples.

    Details:
    https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Sobol_Sampler

    This sampler can introduce obvious grid like artifacts. Consider using the
    HaltonSampler or HammersleySampler instead.
    """

    def __init__(self, **kwargs):
        super(SobolSampler, self).__init__(**kwargs)
        self._sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        upper = math.ceil(num_samples / 2)
        samples = self._sobol_engine.draw(batch_size * upper, dtype=dtype)
        return samples.view(batch_size, -1, 1).to(device=device)[:, :num_samples]

    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        samples = self._sobol_engine.draw(batch_size * num_samples, dtype=dtype)
        return samples.view(batch_size, num_samples, 2).to(device=device)


class HaltonSampler(AbstractSampler):
    """A good low-discrepancy sampling pattern.

    Details:
    https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler

    Overall visually pleasing results with less noticeable noise compared to
    UniformSampler.
    """

    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        samples = create_halton_sequence(
            batch_size * num_samples, dtype=dtype, device=device
        )
        return samples.view(batch_size, num_samples, 1)

    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        samples = create_halton_sequence(
            batch_size * num_samples, dim=2, dtype=dtype, device=device
        )
        return samples.view(batch_size, num_samples, 2)


class HammersleySampler(AbstractSampler):
    """This samples based on a HaltonSequence, but in 2D one of the dimension is
    linearly distributed.

    Details:
    https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler

    Overall visually pleasing results with less noticeable noise compared to
    UniformSampler.

    This sampler only makes sense when the number of samples is fixed (Only request the
    samples once)
    """

    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        samples = create_halton_sequence(
            batch_size * num_samples, dtype=dtype, device=device
        )
        return samples.view(batch_size, num_samples, 1)

    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        samples = create_halton_sequence(
            batch_size * num_samples, dtype=dtype, device=device
        ).view(batch_size, num_samples)

        linspace = (
            torch.linspace(0, 1, num_samples + 2, device=device, dtype=dtype)[1:-1]
            .view(1, num_samples)
            .expand(batch_size, -1)
        )

        return torch.stack((samples, linspace), -1)


class UniformSampler(AbstractSampler):
    """Base Sampling strategy. Just distributes samples uniformly between 0-1.
    Rendering with this sampler introduces a quite noticeable random noise pattern.
    """

    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        return torch.rand(batch_size, num_samples, 1, device=device, dtype=dtype)

    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        return torch.rand(batch_size, num_samples, 2, device=device, dtype=dtype)


class StratifiedSampler(AbstractSampler):
    """Stratified sampling strategy:

    Details:
    https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Stratified_Sampling

    Overall the render result looks slightly better than UniformSampler.
    Still not as good as HaltonSampler.

    If equal_sampling is enabled, the number of samples should be equal to or a multiple
    of the number of cells in the stratified sampling grid.
    """

    def __init__(
        self,
        grid_x: int,
        grid_y: int,
        jitter: bool = True,
        equal_sampling: bool = False,
        **kwargs,
    ):
        super(StratifiedSampler, self).__init__(**kwargs)
        self._x_size = 1 / grid_x
        self._y_size = 1 / grid_y
        self._jitter = jitter
        self._equal_sampling = equal_sampling

        grid = torch.meshgrid(torch.arange(grid_x), torch.arange(grid_y), indexing="xy")

        self.register_buffer(
            "_grid_2d_flat", torch.stack(grid, -1).view(-1, 2), persistent=False
        )
        self.register_buffer(
            "_grid_1d_flat", torch.arange(grid_x).view(-1, 1), persistent=False
        )

        self.register_buffer(
            "_cell_size",
            torch.tensor([self._x_size, self._y_size]).unsqueeze(0),
            persistent=False,
        )

    def _get_grid_idx(
        self, grid, num_samples, device=None
    ) -> Float[Tensor, "num_samples"]:  # noqa: F821
        grid_size = grid.shape[0]
        if self._equal_sampling:
            num_sampling_steps = max(1, math.ceil(num_samples / grid_size))
            samples = []
            for _ in range(num_sampling_steps):
                samples.append(torch.randperm(grid_size, device=device))
            samples = torch.cat(samples, 0)
            return samples[:num_samples]
        else:
            return torch.randint(0, grid_size, (num_samples,))

    def samples_1D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 1"]:
        grid_idxs = self._get_grid_idx(self._grid_1d_flat, num_samples, device=device)
        base_positions = (self._grid_1d_flat[grid_idxs] * self._x_size).view(1, -1, 1)

        offsets = (
            torch.rand(batch_size, num_samples, 1, device=device, dtype=dtype)
            if self._jitter
            else torch.full((batch_size, num_samples, 1), 0.5, device=device)
        ) * self._x_size

        return base_positions + offsets

    def samples_2D_impl(
        self, batch_size: int, num_samples: int, device=None, dtype=None
    ) -> Float[Tensor, "batch num_samples 2"]:
        grid_idxs = self._get_grid_idx(self._grid_2d_flat, num_samples, device=device)
        base_positions = (self._grid_2d_flat[grid_idxs] * self._cell_size).view(
            1, -1, 2
        )

        offsets = (
            torch.rand(batch_size, num_samples, 2, device=device, dtype=dtype)
            if self._jitter
            else torch.full((batch_size, num_samples, 2), 0.5, device=device)
        ) * self._cell_size

        return base_positions + offsets


SAMPLING_STRATEGIES = {
    "uniform": UniformSampler,
    "halton": HaltonSampler,
    "hammersley": HammersleySampler,
    "sobol": SobolSampler,
    "stratified": StratifiedSampler,
}

# Several handy sample warping functions: Math based on:
# https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformlySamplingaHemisphere


def square_to_uniform_sphere(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 3"]:
    sx, sy = sample[:, 0], sample[:, 1]

    z = 1 - 2 * sy
    r = safe_sqrt(1 - z * z)
    phi = 2 * torch.pi * sx
    sin_phi, cos_phi = phi.sin(), phi.cos()
    return torch.stack((r * cos_phi, r * sin_phi, z), -1)


def square_to_uniform_sphere_pdf() -> float:
    return 1 / (4 * torch.pi)


def square_to_uniform_hemisphere(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 3"]:
    sx, sy = sample[:, 0], sample[:, 1]

    r = safe_sqrt(1.0 - sx * sx)

    phi = 2 * torch.pi * sy
    sin_phi, cos_phi = phi.sin(), phi.cos()

    return torch.stack((r * cos_phi, r * sin_phi, sx), -1)


def square_to_uniform_hemisphere_pdf() -> float:
    return 1 / (2 * torch.pi)


def square_to_cosine_hemisphere(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 3"]:
    p = square_to_uniform_disk_concentric(sample)
    px, py = p[:, 0], p[:, 1]
    z = safe_sqrt(1 - px.square() - py.square()).clip(min=1e-10)

    return torch.stack((px, py, z), -1)


def square_to_cosine_hemisphere_pdf(
    local_direction: Float[Tensor, "N 3"],
) -> Float[Tensor, "N 1"]:
    cos_theta = local_direction[..., 2:3].clip(0, 1)
    return 1 / torch.pi * cos_theta


def square_to_uniform_cone(
    cos_cutoff: Union[float, Float[Tensor, "N 1"]],
    sample: Float[Tensor, "N 2"],
) -> Float[Tensor, "N 3"]:
    sx, sy = sample[:, 0], sample[:, 1]

    cos_theta = (1 - sx) + sx * cos_cutoff
    sin_theta = safe_sqrt(1 - cos_theta.square())

    phi = 2 * torch.pi * sy
    sin_phi, cos_phi = phi.sin(), phi.cos()

    return torch.stack((cos_phi * sin_theta, sin_phi * sin_theta, cos_theta), -1)


def square_to_uniform_cone_pdf(
    cos_cutoff: Union[float, Float[Tensor, "*B 1"]],
) -> Union[float, Float[Tensor, "*B 1"]]:
    return 1 / (2 * torch.pi) / (1 - cos_cutoff)


def square_to_uniform_disk_concentric(
    sample: Float[Tensor, "N 2"],
) -> Float[Tensor, "N 2"]:
    sx, sy = sample[:, 0], sample[:, 1]

    r1 = 2 * sx - 1
    r2 = 2 * sy - 1

    rphi = torch.where(
        torch.logical_and(r1 == 0, r2 == 0).unsqueeze(1).expand(-1, 2),
        torch.zeros_like(sample),
        torch.where(
            (r1.square() > r2.square()).unsqueeze(1).expand(-1, 2),
            torch.stack((r1, (torch.pi / 4 * torch.nan_to_num(r2 / r1))), -1),
            torch.stack(
                (r2, (torch.pi / 2 - torch.nan_to_num(r1 / r2) * torch.pi / 4)), -1
            ),
        ),
    )

    r, phi = rphi[:, 0], rphi[:, 1]
    sin_phi, cos_phi = phi.sin(), phi.cos()

    return torch.stack((r * cos_phi, r * sin_phi), -1)


def square_to_uniform_disk_concentric_pdf() -> float:
    return 1 / (torch.pi)


def square_to_uniform_disk(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 2"]:
    sx, sy = sample[:, 0], sample[:, 1]
    r = safe_sqrt(sx)

    phi = 2 * torch.pi * sy
    sin_phi, cos_phi = phi.sin(), phi.cos()

    return torch.stack((cos_phi * r, sin_phi * r), -1)


def square_to_uniform_disk_pdf() -> float:
    return 1 / torch.pi


def square_to_uniform_triangle(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 2"]:
    sx, sy = sample[:, 0], sample[:, 1]
    a = safe_sqrt(1 - sx)
    return torch.stack((1 - a, a * sy), -1)


def interval_to_tent(sample: Float[Tensor, "N"]) -> Float[Tensor, "N"]:  # noqa: F821
    signsample = torch.where(
        sample.unsqueeze(-1).repeat(1, 2) < 0.5,
        torch.stack((torch.ones_like(sample), sample * 2), -1),
        torch.stack((-torch.ones_like(sample), 2 * (sample - 0.5)), -1),
    )

    sign, smpl = signsample[:, 0], signsample[:, 1]
    return sign * (1 - torch.sqrt(smpl))


def square_to_tent(sample: Float[Tensor, "N 2"]) -> Float[Tensor, "N 2"]:
    sx, sy = sample[:, 0], sample[:, 1]
    return torch.stack((interval_to_tent(sx), interval_to_tent(sy)), -1)
