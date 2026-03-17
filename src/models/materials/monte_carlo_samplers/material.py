from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from src.constants import Names, OutputsType
from src.models.materials.pbr_utils import (
    GGXNormalDistributionFunction,
    GGXNormalDistributionFunctionMediumPrecision,
    SmithUE4schlickGGXGeometricShadowing,
)
from src.utils.coordinate_frame import Frame
from src.utils.ops import (
    EPS_DTYPE,
    dot,
    normalize,
    reflect,
    safe_acos,
    safe_sqrt,
    spherical_to_cartesian,
)
from src.utils.sampling_utils import (
    AbstractSampler,
    Samples,
    square_to_cosine_hemisphere,
    square_to_cosine_hemisphere_pdf,
)

from .abstract_sampler import AbstractMonteCarloSampler


class CosineHemisphereMaterialSampler(AbstractMonteCarloSampler):
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


class GGXMaterialSampler(AbstractMonteCarloSampler):
    # Sampling based on a simplified version of: https://agraphicsguynotes.com/posts/sample_microfacet_brdf/

    @dataclass
    class Config(AbstractMonteCarloSampler.Config):
        use_medium_precision: bool = False
        perceptual_roughness: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        if self.cfg.use_medium_precision:
            self.ndf = GGXNormalDistributionFunctionMediumPrecision()
        else:
            self.ndf = GGXNormalDistributionFunction()

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        batch_size, elements = normal.shape[0], normal.shape[1]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=roughness.dtype,
        )
        x, y = (
            samples[..., 0:1].view(batch_size, elements, self.cfg.num_samples, 1),
            samples[..., 1:2].view(batch_size, elements, self.cfg.num_samples, 1),
        )

        rough_ext = (
            roughness.view(batch_size, -1, 1, 1)
            .repeat(1, 1, self.cfg.num_samples, 1)
            .clip(0, 1)
        )
        alpha = rough_ext
        if self.cfg.perceptual_roughness:
            alpha = alpha.square()
        a2_ext = alpha.square()
        normal_ext = normal.view(batch_size, -1, 1, 3).repeat(
            1, 1, self.cfg.num_samples, 1
        )
        view_direction_ext = viewdirs.view(batch_size, -1, 1, 3).repeat(
            1, 1, self.cfg.num_samples, 1
        )

        frame = Frame(normal_ext)

        cos_theta = safe_sqrt(
            (1 - x) / ((a2_ext - 1) * x + 1).clip(EPS_DTYPE[a2_ext.dtype], 1),
            eps=EPS_DTYPE[a2_ext.dtype],
        ).clip(EPS_DTYPE[a2_ext.dtype], 1 - EPS_DTYPE[a2_ext.dtype])
        theta = safe_acos(cos_theta)
        phi = 2 * torch.pi * y

        # Spherical to cartesian (z up)
        local_wm = spherical_to_cartesian(theta, phi)
        world_wm = frame.to_world(local_wm)
        world_wi = reflect(-view_direction_ext, world_wm)
        # We now have the world incident and halfway

        # Construct the PDF
        ndf = self.ndf(
            alpha,
            cos_theta,
            normal_ext,
            world_wm,
            perceptual_roughness=False,  # Already handled
        )
        denominator_incident = 4 * dot(world_wm, view_direction_ext).clip(
            EPS_DTYPE[ndf.dtype], 1
        )
        pdf = (ndf * cos_theta) / denominator_incident

        return Samples(world_wi, pdf)

    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        halfway_vector = normalize(viewdirs.unsqueeze(2) + directions)
        cos_theta = dot(normal.unsqueeze(2), halfway_vector).clip(0, 1)

        ndf = self.ndf(
            roughness.unsqueeze(2),
            cos_theta,
            normal.unsqueeze(2),
            halfway_vector,
            perceptual_roughness=self.cfg.perceptual_roughness,
        )
        denominator_incident = 4 * dot(halfway_vector, viewdirs.unsqueeze(2)).clip(
            EPS_DTYPE[ndf.dtype], 1
        )
        pdf = (ndf * cos_theta) / denominator_incident

        return pdf


class GGXAntitheticMaterialSampler(GGXMaterialSampler):
    def configure(self):
        super().configure()
        if self.cfg.num_samples % 2 == 1:
            raise ValueError("num_samples must be even")

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        batch_size, elements = normal.shape[0], normal.shape[1]

        actual_num_samples = self.cfg.num_samples // 2
        samples = sampler.get_samples_2D(
            batch_size * elements,
            actual_num_samples,
            device=normal.device,
            dtype=roughness.dtype,
        )
        x, y = (
            samples[..., 0:1].view(batch_size, elements, actual_num_samples, 1),
            samples[..., 1:2].view(batch_size, elements, actual_num_samples, 1),
        )

        rough_ext = (
            roughness.view(batch_size, -1, 1, 1)
            .repeat(1, 1, actual_num_samples, 1)
            .clip(0, 1)
        )
        alpha = rough_ext
        if self.cfg.perceptual_roughness:
            alpha = alpha.square()
        a2_ext = alpha.square()
        normal_ext = normal.view(batch_size, -1, 1, 3).repeat(
            1, 1, actual_num_samples, 1
        )
        view_direction_ext = viewdirs.view(batch_size, -1, 1, 3).repeat(
            1, 1, actual_num_samples, 1
        )

        frame = Frame(normal_ext)

        cos_theta = safe_sqrt(
            (1 - x) / ((a2_ext - 1) * x + 1).clip(EPS_DTYPE[a2_ext.dtype], 1),
            eps=EPS_DTYPE[a2_ext.dtype],
        ).clip(EPS_DTYPE[a2_ext.dtype], 1 - EPS_DTYPE[a2_ext.dtype])
        theta = safe_acos(cos_theta)
        phi = 2 * torch.pi * y

        # Spherical to cartesian (z up)
        local_wm = spherical_to_cartesian(theta, phi)
        world_wm = frame.to_world(local_wm)
        mirror_wm = reflect(-world_wm, normal_ext)

        full_wm = torch.cat([world_wm, mirror_wm], dim=-2)
        normal_double_ext = torch.cat([normal_ext, normal_ext], dim=-2)
        view_direction_double_ext = torch.cat(
            [view_direction_ext, view_direction_ext], dim=-2
        )
        alpha_double_ext = torch.cat([alpha, alpha], dim=-2)
        cos_theta_double = torch.cat([cos_theta, cos_theta], dim=-2)
        world_wi = reflect(-view_direction_double_ext, full_wm)
        # We now have the world incident and halfway

        # Construct the PDF
        ndf = self.ndf(
            alpha_double_ext,
            cos_theta_double,
            normal_double_ext,
            full_wm,
            perceptual_roughness=False,  # Already handled
        )
        denominator_incident = 4 * dot(full_wm, view_direction_double_ext).clip(
            EPS_DTYPE[ndf.dtype], 1
        )
        pdf = (ndf * cos_theta_double) / denominator_incident

        return Samples(world_wi, pdf)


class GGXVNDFMaterialSampler(AbstractMonteCarloSampler):
    @dataclass
    class Config(AbstractMonteCarloSampler.Config):
        use_spherical_caps: bool = True
        use_medium_precision: bool = False
        perceptual_roughness: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        if self.cfg.use_medium_precision:
            self.ndf = GGXNormalDistributionFunctionMediumPrecision()
        else:
            self.ndf = GGXNormalDistributionFunction()
        self.geom_shadowing = SmithUE4schlickGGXGeometricShadowing()

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        batch_size, elements = normal.shape[0], normal.shape[1]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=roughness.dtype,
        ).view(batch_size, elements, self.cfg.num_samples, 2)

        normal = normalize(normal)

        batch_size, elements = normal.shape[0], normal.shape[1]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            self.cfg.num_samples,
            device=normal.device,
            dtype=roughness.dtype,
        ).view(batch_size, elements, self.cfg.num_samples, 2)

        rough_ext = (
            roughness.view(batch_size, -1, 1, 1)
            .repeat(1, 1, self.cfg.num_samples, 1)
            .clip(0, 1)
        )
        alpha = rough_ext
        if self.cfg.perceptual_roughness:
            alpha = alpha.square()
        alpha2D = torch.cat([alpha, alpha], dim=-1)  # Same roughness in both directions

        normal_ext = normal.view(batch_size, -1, 1, 3).repeat(
            1, 1, self.cfg.num_samples, 1
        )
        view_direction_ext = viewdirs.view(batch_size, -1, 1, 3).repeat(
            1, 1, self.cfg.num_samples, 1
        )

        frame = Frame(normal_ext)
        local_view = frame.to_local(view_direction_ext)

        # Transform view direction to hemisphere configuration
        Vh = normalize(
            torch.stack(
                [
                    alpha2D[..., 0] * local_view[..., 0],
                    alpha2D[..., 1] * local_view[..., 1],
                    local_view[..., 2],
                ],
                dim=-1,
            )
        )

        if self.cfg.use_spherical_caps:
            # Spherical caps sampling method
            phi = 2.0 * torch.pi * samples[..., 0]
            z = ((1.0 - samples[..., 1]) * (1.0 + Vh[..., 2])) - Vh[..., 2]
            sin_theta = safe_sqrt(torch.clamp(1.0 - z * z, 0.0, 1.0))
            x = sin_theta * torch.cos(phi)
            y = sin_theta * torch.sin(phi)

            # Compute halfway direction
            Nh = torch.stack([x, y, z], dim=-1) + Vh
        else:
            # Original method from "Sampling the GGX Distribution of Visible Normals"
            # Construct orthonormal basis
            lensq = Vh[..., 0].square() + Vh[..., 1].square()  # [1, 669124, 16]
            T1 = torch.where(
                lensq.unsqueeze(-1) > 0,  # [1, 669124, 16, 1]
                torch.stack(
                    [-Vh[..., 1], Vh[..., 0], torch.zeros_like(Vh[..., 0])], dim=-1
                )
                / safe_sqrt(lensq, eps=EPS_DTYPE[lensq.dtype]).unsqueeze(-1),
                torch.ones_like(Vh)
                * torch.tensor([1.0, 0.0, 0.0], device=Vh.device, dtype=Vh.dtype).view(
                    1, 1, 1, 3
                ),
            )
            T2 = torch.cross(Vh, T1, dim=-1)

            # Sample projected area
            r = safe_sqrt(samples[..., 0])
            phi = 2 * torch.pi * samples[..., 1]
            t1 = r * torch.cos(phi)
            t2 = r * torch.sin(phi)
            s = 0.5 * (1 + Vh[..., 2])
            t2 = torch.lerp(safe_sqrt(1 - t1.square()), t2, s)

            # Reproject onto hemisphere
            Nh = (
                t1.unsqueeze(-1) * T1
                + t2.unsqueeze(-1) * T2
                + safe_sqrt(
                    torch.clamp(1 - t1.square() - t2.square(), min=0).unsqueeze(-1)
                )
                * Vh
            )

        # Transform back to ellipsoid configuration
        local_wm = normalize(
            torch.stack(
                [
                    alpha2D[..., 0] * Nh[..., 0],
                    alpha2D[..., 1] * Nh[..., 1],
                    torch.clamp(Nh[..., 2], min=0),
                ],
                dim=-1,
            )
        )

        # Convert to world space
        world_wm = frame.to_world(local_wm)
        world_wi = reflect(-view_direction_ext, world_wm)

        # Compute PDF
        NdotH = dot(normal_ext, world_wm).clip(EPS_DTYPE[alpha.dtype], 1)
        NdotV = dot(normal_ext, view_direction_ext).clip(EPS_DTYPE[alpha.dtype], 1)

        D = self.ndf(
            alpha,
            NdotH,
            normal_ext,
            world_wm,
            perceptual_roughness=False,  # Already handled
        )
        G1 = self.geom_shadowing.partial_smith_function(NdotV, alpha)

        pdf = (D * G1) / (4 * NdotV)

        return Samples(world_wi, pdf)

    def _pdf_impl(
        self,
        outputs: OutputsType,
        directions: Float[Tensor, "B S num_samples 3"],
    ) -> Float[Tensor, "B S num_samples 1"]:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        normal = normalize(normal)

        halfway_vector = normalize(viewdirs.unsqueeze(2) + directions)

        NdotH = dot(normal.unsqueeze(2), halfway_vector).clip(
            EPS_DTYPE[roughness.dtype], 1
        )
        NdotV = dot(normal.unsqueeze(2), viewdirs.unsqueeze(2)).clip(
            EPS_DTYPE[roughness.dtype], 1
        )

        alpha = roughness
        if self.cfg.perceptual_roughness:
            alpha = alpha.square()
        D = self.ndf(
            alpha.unsqueeze(2),
            NdotH,
            normal.unsqueeze(2),
            halfway_vector,
            perceptual_roughness=False,  # Already handled
        )
        G1 = self.geom_shadowing.partial_smith_function(NdotV, alpha.unsqueeze(2))

        pdf = (D * G1) / (4 * NdotV)

        return pdf


class GGXVNDFAntitheticMaterialSampler(GGXVNDFMaterialSampler):
    def configure(self):
        super().configure()
        if self.cfg.num_samples % 2 != 0:
            raise ValueError("num_samples must be even")

    def _generate_samples_impl(
        self,
        outputs: OutputsType,
        sampler: AbstractSampler,
    ) -> Samples:
        normal = normalize(outputs[Names.SHADING_NORMAL.add_suffix(self.cfg.suffix)])
        roughness = outputs[Names.ROUGHNESS.add_suffix(self.cfg.suffix)]
        viewdirs = outputs[Names.VIEW_DIRECTION.add_suffix(self.cfg.suffix)]

        # ------------------------------------------------------------
        # 1) We only draw half as many seeds:
        # ------------------------------------------------------------
        batch_size, elements = normal.shape[0], normal.shape[1]
        half_num = self.cfg.num_samples // 2

        # shape = [batch_size * elements, half_num, 2] -> then reshaped to [B, E, half_num, 2]
        samples = sampler.get_samples_2D(
            batch_size * elements,
            half_num,
            device=normal.device,
            dtype=roughness.dtype,
        ).view(batch_size, elements, half_num, 2)

        # Prepare repeated buffers in [B, E, half_num, *]
        rough_ext = (
            roughness.view(batch_size, elements, 1, 1)
            .repeat(1, 1, half_num, 1)
            .clamp(0, 1)
        )
        alpha = rough_ext.square() if self.cfg.perceptual_roughness else rough_ext
        alpha2D = torch.cat([alpha, alpha], dim=-1)  # same alpha in x,y

        normal_ext = (
            normalize(normal).view(batch_size, elements, 1, 3).repeat(1, 1, half_num, 1)
        )
        view_direction_ext = viewdirs.view(batch_size, elements, 1, 3).repeat(
            1, 1, half_num, 1
        )

        # Build local frame
        frame = Frame(normal_ext)
        local_view = frame.to_local(view_direction_ext)

        # ------------------------------------------------------------
        # 2) Sample one visible half-vector h per seed
        #    using the same VNDF logic as the parent class
        # ------------------------------------------------------------
        # Step 2a) "Transform" local_view -> 'Vh'
        Vh = normalize(
            torch.stack(
                [
                    alpha2D[..., 0] * local_view[..., 0],
                    alpha2D[..., 1] * local_view[..., 1],
                    local_view[..., 2],
                ],
                dim=-1,
            )
        )

        if self.cfg.use_spherical_caps:
            phi = 2.0 * torch.pi * samples[..., 0]
            z = ((1.0 - samples[..., 1]) * (1.0 + Vh[..., 2])) - Vh[..., 2]
            sin_theta = safe_sqrt(torch.clamp(1.0 - z * z, 0.0, 1.0))
            x = sin_theta * torch.cos(phi)
            y = sin_theta * torch.sin(phi)
            Nh = torch.stack([x, y, z], dim=-1) + Vh
        else:
            # "Original" VNDF sampling code
            lensq = Vh[..., 0].square() + Vh[..., 1].square()
            T1 = torch.where(
                lensq.unsqueeze(-1) > 0,
                torch.stack(
                    [-Vh[..., 1], Vh[..., 0], torch.zeros_like(Vh[..., 0])], dim=-1
                )
                / safe_sqrt(lensq, eps=EPS_DTYPE[lensq.dtype]).unsqueeze(-1),
                # fallback if lensq=0
                torch.ones_like(Vh)
                * torch.tensor([1.0, 0.0, 0.0], device=Vh.device, dtype=Vh.dtype).view(
                    1, 1, 1, 3
                ),
            )
            T2 = torch.cross(Vh, T1, dim=-1)

            r = safe_sqrt(samples[..., 0])
            phi = 2.0 * torch.pi * samples[..., 1]
            t1 = r * torch.cos(phi)
            t2 = r * torch.sin(phi)
            s = 0.5 * (1.0 + Vh[..., 2])
            t2 = torch.lerp(safe_sqrt(1.0 - t1.square()), t2, s)

            Nh = (
                t1.unsqueeze(-1) * T1
                + t2.unsqueeze(-1) * T2
                + safe_sqrt(
                    torch.clamp(1.0 - t1.square() - t2.square(), min=0).unsqueeze(-1)
                )
                * Vh
            )

        local_wm = normalize(
            torch.stack(
                [
                    alpha2D[..., 0] * Nh[..., 0],
                    alpha2D[..., 1] * Nh[..., 1],
                    torch.clamp(Nh[..., 2], min=0),
                ],
                dim=-1,
            )
        )
        # -> shape [B, E, half_num, 3]
        world_wm = frame.to_world(local_wm)  # [B, E, half_num, 3]

        # ------------------------------------------------------------
        # 3) Create the mirrored half-vector h' by reflecting h across n
        # ------------------------------------------------------------
        mirror_wm = reflect(-world_wm, normal_ext)  # [B, E, half_num, 3]

        # Concatenate them => 2 x half_num directions
        full_wm = torch.cat([world_wm, mirror_wm], dim=-2)
        normal_double_ext = torch.cat([normal_ext, normal_ext], dim=-2)
        viewdir_double_ext = torch.cat([view_direction_ext, view_direction_ext], dim=-2)

        # ------------------------------------------------------------
        # 4) Reflect the view direction around both half-vectors
        #    to get the final directions
        # ------------------------------------------------------------
        world_wi = reflect(-viewdir_double_ext, full_wm)  # [B, E, 2*half_num, 3]

        # ------------------------------------------------------------
        # 5) Compute PDF for each final direction
        # ------------------------------------------------------------
        #   D * G1 / (4 * NdotV)
        NdotH = dot(normal_double_ext, full_wm).clamp(EPS_DTYPE[alpha.dtype], 1)
        NdotV = dot(normal_double_ext, viewdir_double_ext).clamp(
            EPS_DTYPE[alpha.dtype], 1
        )

        D = self.ndf(
            torch.cat([alpha, alpha], dim=-2),  # same alpha for both sets
            NdotH,
            normal_double_ext,
            full_wm,
            perceptual_roughness=False,  # already handled
        )
        G1 = self.geom_shadowing.partial_smith_function(
            NdotV, torch.cat([alpha, alpha], dim=-2)
        )

        pdf = (D * G1) / (4.0 * NdotV)  # [B, E, 2*half_num, 1 or scalar]

        return Samples(world_wi, pdf)
