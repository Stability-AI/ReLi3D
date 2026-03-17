from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

import src
from src.constants import FieldName, Names, OutputsType
from src.models.illumination.env_map_parametrization.octahedral import (
    OctahedralEnvRepresentationTexture,
)
from src.models.illumination.env_map_parametrization.spherical import (
    SphericalEnvRepresentationTexture,
)
from src.models.illumination.env_map_parametrization.texture import (
    BilinearTextureAccess,
)
from src.utils.color_space import srgb_to_linear
from src.utils.ops import EPS_DTYPE, dot, mix, normalize
from src.utils.sampling_utils import (
    SAMPLING_STRATEGIES,
    CranleyPattersonRotationMutationStrategy,
    Samples,
)
from src.utils.typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseMaterial
from .monte_carlo_samplers.illumination import (
    PiecewiseDistributionEnvironmentSkySampler,
)
from .pbr_utils import FRESNEL_FUNCTIONS, GEO_SHADOWING_FUNCTIONS, NDF_FUNCTIONS

ENV_PARAMETRIZATION = {
    "spherical": SphericalEnvRepresentationTexture,
    "octahedral": OctahedralEnvRepresentationTexture,
}


class CookTorranceMaterial(torch.nn.Module):
    """Slightly more expensive Cook-Torrance material model with GGX distribution functions."""

    def __init__(
        self,
        *args,
        is_basecolor_metallic: bool = True,
        base_reflectivity: float = 0.04,
        normal_distribution_function: str = "ggx",
        geometric_shadowing: str = "smith_ue4schlick_ggx",
        fresnel_term: str = "schlick",
        perceptual_roughness: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._is_basecolor_metallic = is_basecolor_metallic
        self._base_reflectivity = base_reflectivity
        self.fresnel_term = FRESNEL_FUNCTIONS[fresnel_term]()
        self.normal_distribution_function = NDF_FUNCTIONS[
            normal_distribution_function
        ]()
        self.geometric_shadowing = GEO_SHADOWING_FUNCTIONS[geometric_shadowing]()
        self.perceptual_roughness = perceptual_roughness

    def get_diffuse_specular_roughness(
        self,
        roughness: Float[Tensor, "*B 1"],
        basecolor: Optional[Float[Tensor, "*B 3"]],
        metallic: Optional[Float[Tensor, "*B 1"]] = None,
        diffuse: Optional[Float[Tensor, "*B 3"]] = None,
        specular: Optional[Float[Tensor, "*B 3"]] = None,
    ) -> Tuple[Float[Tensor, "*B 3"], Float[Tensor, "*B 3"], Float[Tensor, "*B 1"]]:
        """Returns the linear diffuse and specular color as well as the roughness.
        This automatically extracts the parameter based on the selected parametrization
        """
        roughness = roughness.clip(0.05, 1)

        if self._is_basecolor_metallic:
            assert metallic is not None

            basecolor_lin = srgb_to_linear(basecolor)

            diffuse_color = basecolor_lin * (
                1 - metallic
            )  # Only diffuse is metallic is 0
            # Interpolate between 0.04 base reflectivity where non-metallic
            # and specular color (from basecolor)
            specular_color = mix(
                torch.ones_like(basecolor_lin) * self._base_reflectivity,
                basecolor_lin,
                metallic,
            )
        else:
            diffuse_color = srgb_to_linear(diffuse)
            specular_color = srgb_to_linear(specular)

        if self.perceptual_roughness:
            roughness = roughness.square()

        return diffuse_color, specular_color, roughness

    def forward(
        self,
        light_direction: Float[Tensor, "*B 3"],  # Light direction
        view_direction: Float[Tensor, "*B 3"],  # View direction
        normal: Float[Tensor, "*B 3"],
        roughness: Float[Tensor, "*B 1"],
        basecolor: Optional[Float[Tensor, "*B 3"]] = None,
        metallic: Optional[Float[Tensor, "*B 1"]] = None,
        diffuse: Optional[Float[Tensor, "*B 3"]] = None,
        specular: Optional[Float[Tensor, "*B 3"]] = None,
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        diffuse_color, specular_color, roughness = self.get_diffuse_specular_roughness(
            basecolor=basecolor,
            metallic=metallic,
            roughness=roughness,
            diffuse=diffuse,
            specular=specular,
        )

        halfway_vector = normalize(light_direction + view_direction)

        # Pre-calculations
        ndl = dot(normal, light_direction).clip(min=0, max=1)
        ndv = dot(normal, view_direction).clip(min=0, max=1)
        ndh = dot(normal, halfway_vector).clip(
            EPS_DTYPE[normal.dtype], 1 - EPS_DTYPE[normal.dtype]
        )
        vdh = dot(view_direction, halfway_vector).clip(0, 1)

        # Diffuse evaluation: Divide by pi for the energy conservation
        # Plain lambertian
        diffuse = diffuse_color / torch.pi

        # Some helpful pre-calculations.
        D = self.normal_distribution_function(
            roughness,
            ndh,
            normal,
            halfway_vector,
            perceptual_roughness=False,  # Already handled
        )
        F = self.fresnel_term(specular_color, vdh)
        G = self.geometric_shadowing(
            ndv.clip(EPS_DTYPE[ndv.dtype]),
            ndl.clip(EPS_DTYPE[ndl.dtype]),
            vdh,
            ndh,
            roughness,
            perceptual_roughness=False,  # Already handled
        )

        specular = (D * F * G) / (4 * ndl * ndv).clip(EPS_DTYPE[ndl.dtype])

        # The ratio of reflected light is defined in the fresnel term
        # This ensures energy conservation
        kD = (1 - F).clip(min=0, max=1)
        ret = kD * diffuse + specular

        return ret.clip(0)


class MultipleImportanceMonteCarloEnvironmentShader(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        detached_sampling: bool = True

        sampling_stategies: List[Dict[str, Any]] = field(default_factory=list)

        sampler: str = "uniform"
        sample_rotation: bool = False
        sample_rotation_scale: float = 0.025

        illumination_representation: str = "spherical"

        use_power_heuristic: bool = False
        radiance_clamping_upper_limit: Optional[float] = None

        visibility_tester_cls: Optional[str] = None
        visibility_tester: dict = field(default_factory=dict)
        visibility_fade_steps: int = 0

        is_basecolor_metallic: bool = True
        base_reflectivity: float = 0.04

        perceptual_roughness: bool = True
        ndf: str = "ggx"
        geo_shadowing: str = "smith_ue4schlick_ggx"
        fresnel: str = "schlick"

    def configure(self) -> None:
        super().configure()

        self.material = CookTorranceMaterial(
            is_basecolor_metallic=self.cfg.is_basecolor_metallic,
            base_reflectivity=self.cfg.base_reflectivity,
            normal_distribution_function=self.cfg.ndf,
            geometric_shadowing=self.cfg.geo_shadowing,
            fresnel_term=self.cfg.fresnel,
            perceptual_roughness=self.cfg.perceptual_roughness,
        )

        self.requires_distribution_2d = False

        sample_strategies = []
        samples_per_strategy = []
        for strategy in self.cfg.sampling_stategies:
            strat = src.initialize_instance(
                strategy["cls"],
                strategy["kwargs"] | {"detached_strategy": self.cfg.detached_sampling},
            )
            sample_strategies.append(strat)
            print(strat, strat.cfg)
            samples_per_strategy.append(strat.cfg.num_samples)
            if isinstance(strat, PiecewiseDistributionEnvironmentSkySampler):
                self.requires_distribution_2d = True

        self.total_samples = sum(samples_per_strategy)
        self.samples_per_strategy = samples_per_strategy
        self.sample_strategies = torch.nn.ModuleList(sample_strategies)

        self.sampler = SAMPLING_STRATEGIES[self.cfg.sampler](
            mutation_strategy=(
                CranleyPattersonRotationMutationStrategy(self.cfg.sample_rotation_scale)
                if self.cfg.sample_rotation
                else None
            )
        )
        self.texture_sampler = BilinearTextureAccess()
        self.env_parametrization = ENV_PARAMETRIZATION[
            self.cfg.illumination_representation
        ]()

        self.visibility_tester = None
        if self.cfg.visibility_tester_cls is not None:
            class_to_find = src.find(self.cfg.visibility_tester_cls)
            self.visibility_tester = class_to_find(
                class_to_find.Config(**self.cfg.visibility_tester)
            )

    @property
    def requires_full_images(self):
        return True

    def consumed_keys(self) -> Set[FieldName]:
        return (
            {
                Names.SHADING_NORMAL,
                Names.GEOMETRY_NORMAL,
                Names.SURFACE_NORMAL,
                Names.POSITION,
                Names.ROUGHNESS,
                Names.VIEW_DIRECTION,
                Names.ENV_MAP,
                Names.VISIBLE_RAYS,
                Names.HEIGHT,
                Names.WIDTH,
            }
            | (
                {Names.BASECOLOR, Names.METALLIC}
                if self.cfg.is_basecolor_metallic
                else {Names.DIFFUSE, Names.SPECULAR}
            )
            | (
                self.visibility_tester.consumed_keys()
                if self.visibility_tester is not None
                else set()
            )
        )

    def draw_all_samples(
        self,
        outputs: OutputsType,
        distribution: Optional[OutputsType] = None,
    ):
        ret_smpls = []
        for strategy in self.sample_strategies:
            ret_smpls.append(
                strategy.generate_samples(
                    outputs=(outputs | distribution)
                    if distribution is not None
                    else outputs,
                    sampler=self.sampler,
                )
            )

        return ret_smpls

    def _get_radiance(
        self,
        outputs: OutputsType,
        visibility_fade_value: float = 0.0,
    ) -> Tuple[
        Float[Tensor, "B S N 3"], Float[Tensor, "B S N 3"], Float[Tensor, "B S N 3"]
    ]:
        elements = outputs[Names.LIGHT_DIRECTION.ray_samples.integration_samples].shape[
            1
        ]
        num_samples = outputs[
            Names.LIGHT_DIRECTION.ray_samples.integration_samples
        ].shape[2]

        if self.cfg.is_basecolor_metallic:
            base_color = outputs[Names.BASECOLOR.ray_samples.integration_samples].view(
                1, -1, 3
            )
            metallic = outputs[Names.METALLIC.ray_samples.integration_samples].view(
                1, -1, 1
            )
            diffuse = None
            specular = None
        else:
            diffuse = outputs[Names.DIFFUSE.ray_samples.integration_samples].view(
                1, -1, 3
            )
            specular = outputs[Names.SPECULAR.ray_samples.integration_samples].view(
                1, -1, 3
            )
            base_color = None
            metallic = None

        # Eval brdf
        brdf_eval = self.material(
            view_direction=outputs[
                Names.VIEW_DIRECTION.ray_samples.integration_samples
            ].view(1, -1, 3),
            light_direction=outputs[
                Names.LIGHT_DIRECTION.ray_samples.integration_samples
            ].view(1, -1, 3),
            normal=outputs[Names.SHADING_NORMAL.ray_samples.integration_samples].view(
                1, -1, 3
            ),
            roughness=outputs[Names.ROUGHNESS.ray_samples.integration_samples].view(
                1, -1, 1
            ),
            basecolor=base_color,
            metallic=metallic,
            diffuse=diffuse,
            specular=specular,
        ).view(1, elements, num_samples, 3)

        env_eval_coordinates = self.env_parametrization.coordinate_from_direction(
            outputs[Names.LIGHT_DIRECTION.ray_samples.integration_samples]
        )
        env_eval = (
            self.texture_sampler.sample(
                outputs[Names.ENV_MAP].permute(0, 3, 1, 2),
                env_eval_coordinates,
            )
            .view(1, 3, elements, num_samples)
            .permute(0, 2, 3, 1)
        )  # B S N 3

        if self.visibility_tester is not None:
            vis = self.visibility_tester(outputs)[Names.VISIBILITY]
            visibility = (
                vis.view(1, elements, num_samples, 1) + visibility_fade_value
            ).clip(0, 1)
        else:
            visibility = torch.ones(
                1, elements, num_samples, 1, device=self.device, dtype=torch.float32
            )

        # Combine
        ndl = dot(
            outputs[Names.SHADING_NORMAL.ray_samples.integration_samples],
            outputs[Names.LIGHT_DIRECTION.ray_samples.integration_samples],
        )
        illumination_eval = env_eval * visibility * ndl.clip(0, 1)
        radiance = brdf_eval * illumination_eval
        return radiance, illumination_eval, visibility

    def apply_mc_sum(
        self,
        radiance: Float[Tensor, "B S N C"],
        pdf: Float[Tensor, "B S N 1"],
        eps=None,
    ) -> Float[Tensor, "B S C"]:
        if eps is None:
            eps = EPS_DTYPE[radiance.dtype]

        radiance_cleaned = torch.where(
            pdf.expand(-1, -1, -1, radiance.shape[-1]) < eps,
            torch.zeros_like(radiance),
            radiance,
        )
        radiance_weighted = radiance_cleaned / pdf.clip(eps)

        if (
            self.cfg.radiance_clamping_upper_limit is not None
            and self.cfg.radiance_clamping_upper_limit > 0
        ):
            radiance_weighted = radiance_weighted.clip(
                max=self.cfg.radiance_clamping_upper_limit
            )

        return radiance_weighted.mean(dim=2)

    def cross_evaluate_pdfs(
        self,
        samples: list[Samples],
        outputs: OutputsType,
        distribution: Optional[OutputsType] = None,
    ):
        # And weight them based on the selected heuristic
        mis_func = (
            self.power_heuristic_weight
            if self.cfg.use_power_heuristic
            else self.balance_heuristic_weight
        )

        _batch_size, elements = outputs[Names.SHADING_NORMAL.ray_samples].shape[:2]

        weights = []
        for i, main_samples_taken in enumerate(self.samples_per_strategy):
            pdfs = []
            num_samples = []

            cur_samples = samples[i]
            samples_flat = cur_samples.sample_direction.flatten(2, -2)
            main_pdf = cur_samples.pdf.flatten(2, -2)  # B S N 1
            for j, (strategy, cross_evaluate_samples_taken) in enumerate(
                zip(self.sample_strategies, self.samples_per_strategy)
            ):
                # Only perform cross evaluations between the strategies
                if i == j:
                    continue

                # Take only the required amount of samples for the evaluation (matching the shape of samples_flat)
                pdfs.append(
                    strategy.pdf(
                        outputs=(outputs | distribution)
                        if distribution is not None
                        else outputs,
                        directions=samples_flat.view(1, -1, main_samples_taken, 3),
                    ).view(1, elements, main_samples_taken, 1)
                )
                num_samples.append(cross_evaluate_samples_taken)

            weights.append(mis_func(main_pdf, main_samples_taken, pdfs, num_samples))
        return weights

    def forward_impl(
        self, outputs: OutputsType, shade_ray_samples: bool = False
    ) -> OutputsType:
        if outputs[Names.ENV_MAP].ndim == 3:
            outputs[Names.ENV_MAP] = (
                outputs[Names.ENV_MAP]
                .unsqueeze(0)
                .repeat(outputs[Names.BATCH_SIZE], 1, 1, 1)
            )
        illumination = outputs[Names.ENV_MAP]
        batch_size = outputs[Names.BATCH_SIZE]

        distribution = None
        if self.requires_distribution_2d:
            distribution = [
                PiecewiseDistributionEnvironmentSkySampler.build_distribution(
                    illumination[i].unsqueeze(0)
                )
                for i in range(batch_size)
            ]

        visibility_fade_value = 0.0
        if self.cfg.visibility_fade_steps > 0:
            visibility_fade_value = 1 - min(
                max(outputs[Names.GLOBAL_STEP] / self.cfg.visibility_fade_steps, 0.0),
                1.0,
            )

        # Loop over batch size
        out = {}
        for i in range(batch_size):
            current_batch = {}
            current_batch.update(outputs)
            current_batch.update(
                {
                    k: v[i].unsqueeze(0)
                    if isinstance(v, torch.Tensor)
                    else (v[i] if isinstance(v, list) else v)
                    for k, v in current_batch.items()
                    if not k.is_meta_data
                }
            )
            # Mask non visible rays
            vis_rays = current_batch[Names.VISIBLE_RAYS]
            current_batch.update(
                {
                    k.add_suffix(
                        "ray-samples"
                        if isinstance(v, torch.Tensor)
                        and v.shape[:-1] == vis_rays.shape
                        else ""
                    ): v[vis_rays].unsqueeze(0)
                    if isinstance(v, torch.Tensor) and v.shape[:-1] == vis_rays.shape
                    else v
                    for k, v in current_batch.items()
                    if not k.is_meta_data and k != Names.VISIBLE_RAYS
                }
            )

            # Draw samples from each strategy
            samples_per_strategy = self.draw_all_samples(
                current_batch,
                distribution[i] if distribution is not None else None,
            )

            # Evaluate the drawn samples from each strategy with each other and find the weighting
            weights_per_strategy = self.cross_evaluate_pdfs(
                samples_per_strategy,
                current_batch,
                distribution[i] if distribution is not None else None,
            )

            # Query the radiance once in a big batch
            all_samples = torch.cat(
                [s.sample_direction for s in samples_per_strategy], 2
            )

            current_batch[Names.LIGHT_DIRECTION.ray_samples.integration_samples] = (
                all_samples
            )
            num_samples = current_batch[
                Names.LIGHT_DIRECTION.ray_samples.integration_samples
            ].shape[2]

            current_batch.update(
                {
                    k.integration_samples: v.unsqueeze(-2).repeat_interleave(
                        num_samples, dim=-2
                    )
                    for k, v in current_batch.items()
                    if not k.is_integration_sample and k.is_raysample
                }
            )

            # This is faster and also more correct when using pre-filtered environment mipmap aggregations.
            all_radiance_illumination_visibility = self._get_radiance(
                current_batch,
                visibility_fade_value,
            )

            # Then extract the radiance and visibility for each strategy
            num_samples_per_strategy = [
                s.sample_direction.shape[2] for s in samples_per_strategy
            ]
            (
                radiance_per_strategy,
                illumination_per_strategy,
                visibility_per_strategy,
            ) = [
                torch.split(r, num_samples_per_strategy, dim=2)
                for r in all_radiance_illumination_visibility
            ]

            renders = []
            illuminations = []
            visibilities = []
            for radiance, illum, visibility, samples, weights in zip(
                radiance_per_strategy,
                illumination_per_strategy,
                visibility_per_strategy,
                samples_per_strategy,
                weights_per_strategy,
            ):
                radiance = torch.nan_to_num(radiance)
                illum = torch.nan_to_num(illum)
                weights = torch.nan_to_num(weights)
                renders.append(
                    self.apply_mc_sum(radiance * weights.detach(), samples.pdf)
                )
                illuminations.append(
                    self.apply_mc_sum(illum * weights.detach(), samples.pdf)
                )
                visibilities.append((visibility * weights.detach()).mean(dim=2))

            render_summed = torch.nan_to_num(sum(renders))
            illumination_summed = torch.nan_to_num(sum(illuminations))
            visibility_summed = torch.nan_to_num(sum(visibilities))
            out[Names.RADIANCE.ray_samples] = out.get(
                Names.RADIANCE.ray_samples, []
            ) + [render_summed]
            out[Names.VISIBILITY.ray_samples] = out.get(
                Names.VISIBILITY.ray_samples, []
            ) + [visibility_summed]
            out[Names.IRRADIANCE.ray_samples] = out.get(
                Names.IRRADIANCE.ray_samples, []
            ) + [illumination_summed]

        out = {k: torch.cat([c.squeeze(0) for c in v], dim=0) for k, v in out.items()}

        # Restore the shape of the outputs
        ret = {}
        for k in out.keys():
            batch_size = outputs[Names.BATCH_SIZE]
            n_input = outputs.get(Names.VIEW_SIZE, None)
            height = outputs[Names.HEIGHT]
            width = outputs[Names.WIDTH]

            shape = [batch_size]
            if n_input is not None:
                shape.append(n_input)
            shape.append(height)
            shape.append(width)

            def restore_full(v):
                full_img = torch.zeros(
                    *shape,
                    v.shape[-1],
                    device=v.device,
                    dtype=v.dtype,
                )
                full_img[outputs[Names.VISIBLE_RAYS]] = v.view(-1, v.shape[-1])
                return full_img

            base, _ = k.pop_suffix()
            ret[base] = restore_full(out[k])
        return ret

    def balance_heuristic_weight(
        self,
        main_pdf: Float[Tensor, "*B 1"],
        main_samples: int,
        other_pdf: list[Float[Tensor, "*B 1"]],
        other_samples: list[int],
    ) -> Float[Tensor, "*B 1"]:
        main = main_pdf * main_samples

        others = []
        for pdf, samples in zip(other_pdf, other_samples):
            others.append(pdf * samples)
        other_sum = sum(others)

        # Can only be NaN if all evaluates to 0. The weight should then also be 0
        weight = torch.nan_to_num(main / (main + other_sum))
        return weight

    def power_heuristic_weight(
        self,
        main_pdf: Float[Tensor, "*B 1"],
        main_samples: int,
        other_pdf: list[Float[Tensor, "*B 1"]],
        other_samples: list[int],
    ) -> Float[Tensor, "*B 1"]:
        # Use log space to handle large numbers better
        log_main = torch.log(main_pdf.clip(1e-6, 1e4)) + torch.log(
            torch.tensor(main_samples)
        )
        log_main = 2 * log_main  # Square in log space

        log_others = []
        for pdf, samples in zip(other_pdf, other_samples):
            log_pdf = torch.log(pdf.clip(1e-6, 1e4)) + torch.log(torch.tensor(samples))
            log_others.append(2 * log_pdf)  # Square in log space

        # Convert back from log space
        main = torch.exp(log_main)
        others = [torch.exp(log_o) for log_o in log_others]
        other_sum = sum(others)

        weight = main / (main + other_sum + 1e-6)
        return weight.clip(0, 1)

    def export_impl(
        self,
        outputs: OutputsType,
    ) -> OutputsType:
        roughness = outputs[Names.ROUGHNESS]
        base = {Names.ROUGHNESS: roughness}
        if self.cfg.is_basecolor_metallic:
            base[Names.BASECOLOR] = outputs[Names.BASECOLOR]
            base[Names.METALLIC] = outputs[Names.METALLIC]
        else:
            base[Names.DIFFUSE] = outputs[Names.DIFFUSE]
            base[Names.SPECULAR] = outputs[Names.SPECULAR]
        return base
