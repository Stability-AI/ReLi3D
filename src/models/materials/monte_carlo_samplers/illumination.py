import torch
from jaxtyping import Float
from sampler import Distribution2D
from torch import Tensor

from src.constants import Names, OutputsType
from src.utils.color_space import linear_rgb_to_luminance
from src.utils.coordinate_frame import Frame
from src.utils.ops import cartesian_to_spherical, normalize, spherical_to_cartesian
from src.utils.sampling_utils import (
    AbstractSampler,
    Samples,
    square_to_cosine_hemisphere,
    square_to_cosine_hemisphere_pdf,
    square_to_uniform_sphere,
    square_to_uniform_sphere_pdf,
)

from .abstract_sampler import AbstractMonteCarloSampler


class UniformSphereEnvironmentSampler(AbstractMonteCarloSampler):
    """Performs a uniform spherical sampling of the environment"""

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)]
        batch_size, elements = normal.shape[0], normal.shape[1]

        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=normal.dtype,
        ).view(-1, 2)

        sphere_samples = square_to_uniform_sphere(samples).view(
            batch_size, elements, self.cfg.num_samples, 3
        )  # B, num_samples, 3

        pdf = square_to_uniform_sphere_pdf() * torch.ones_like(sphere_samples[..., :1])

        return Samples(sphere_samples, pdf)

    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        return square_to_uniform_sphere_pdf() * torch.ones_like(directions[..., :1])


class CosineHemisphereEnvironmentSampler(AbstractMonteCarloSampler):
    """Performs a cosine weighted heimispherical sampling of the environment"""

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])

        batch_size, elements = normal.shape[0], normal.shape[1]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=normal.dtype,
        ).view(-1, 2)

        normal_ext = normal.unsqueeze(2)

        # It's not fully random but something slightly smarter (cosine hemisphere)
        cosine_hemisphere = square_to_cosine_hemisphere(samples).view(
            batch_size, elements, self.cfg.num_samples, 3
        )  # B, num_samples, 3

        pdf = square_to_cosine_hemisphere_pdf(cosine_hemisphere)

        # Now we have a Hemisphere facing z up
        frame = Frame(normal_ext)
        oriented_samples = frame.to_world(cosine_hemisphere)

        return Samples(oriented_samples, pdf)

    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        frame = Frame(normal.unsqueeze(2))
        local_dir = frame.to_local(directions)

        return square_to_cosine_hemisphere_pdf(local_dir)


class PiecewiseDistributionEnvironmentSkySampler(AbstractMonteCarloSampler):
    """Performs a piecewise distribution sampling of the environment"""

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        distribution = outputs[Names.ENV_SAMPLING_DISTRIBUTION]
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])

        batch_size, elements = normal.shape[0], normal.shape[1]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=normal.dtype,
        ).view(batch_size, -1, 2)

        env_samples = distribution.sample_continuous(
            samples
        )  # Sample coords are y, x with shape B, S*N, 2
        # Samples are already in normed coordinates (0 to 1)

        # Convert to spherical
        env_continuous_location, env_discrete_location, env_pdf = env_samples
        spherical = env_continuous_location * torch.tensor(
            [torch.pi, torch.pi * 2], device=env_continuous_location.device
        ).view(1, 1, 2)

        theta = spherical[..., 0:1]
        phi = spherical[..., 1:2]
        sinTheta = theta.sin().clip(0, 1)

        # Cartesian directions
        directions = spherical_to_cartesian(theta, phi)  # B, S*N, 3

        pdf_normed = env_pdf / (
            2 * torch.pi * torch.pi * sinTheta.clip(min=1e-6)
        )  # B, S*N, 1
        pdf = torch.where(sinTheta <= 1e-6, torch.zeros_like(env_pdf), pdf_normed)

        return Samples(
            directions.view(batch_size, elements, self.cfg.num_samples, 3),
            pdf.view(batch_size, elements, self.cfg.num_samples, 1),
        )

    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        distribution = outputs[Names.ENV_SAMPLING_DISTRIBUTION]
        batch_size, elements, samples = (
            directions.shape[0],
            directions.shape[1],
            directions.shape[2],
        )

        directions = normalize(directions)
        theta, phi = cartesian_to_spherical(directions)

        y_norm = theta / torch.pi
        x_norm = phi / (2 * torch.pi)

        coord_normed = torch.cat((y_norm, x_norm), -1).clip(0, 1)

        sample_pdf = distribution.discrete_pdf_normed_location(
            coord_normed.view(batch_size, -1, 2)
        )
        sinTheta = theta.sin().view(batch_size, -1, 1).clip(0, 1)

        pdf = torch.where(
            sinTheta <= 1e-6,
            torch.zeros_like(sample_pdf),
            sample_pdf / (2 * torch.pi * torch.pi * sinTheta.clip(min=1e-6)),
        )

        return pdf.view(batch_size, elements, samples, 1)

    @classmethod
    def build_distribution(
        cls, illumination_image: Float[Tensor, "B H W 3"]
    ) -> OutputsType:
        device = illumination_image.device

        if illumination_image.ndim == 3:
            illumination_image = illumination_image.unsqueeze(0)

        thetas, _ = torch.meshgrid(
            torch.linspace(0, torch.pi, illumination_image.shape[1], device=device),
            torch.linspace(
                0, torch.pi * 2, illumination_image.shape[2] + 1, device=device
            )[
                :-1
            ],  # Full 2*pi would be same as 0 - Create one extra point and ignore it
            indexing="ij",
        )

        env_map_luminance = linear_rgb_to_luminance(illumination_image).view(
            *illumination_image.shape[:-1]
        )
        env_map_sin_theta_weighted = env_map_luminance * thetas.unsqueeze(0).sin().clip(
            0, 1
        )

        return {
            Names.ENV_SAMPLING_DISTRIBUTION: Distribution2D(env_map_sin_theta_weighted)
        }
