import abc

import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.ops import EPS_DTYPE, safe_sqrt

# Implementations based on: http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

# ----------------------------------------------------------------------------
# NDFs
# ----------------------------------------------------------------------------


class AbstractNormalDistributionFunction(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def normal_distribution_function(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
    ) -> Float[Tensor, "*B 1"]:
        pass

    def forward(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
        perceptual_roughness: bool = True,
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the normal distribution function.

        Args:
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.
            perceptual_roughness (bool, optional): If True, the roughness is assumed to be perceptual.
                Otherwise, it is assumed to be linear.

        Returns:
            Float[Tensor, "*B 1"]: The normal distribution function.
        """
        if perceptual_roughness:
            roughness = roughness.square()

        return self.normal_distribution_function(
            roughness, normal_dot_halfway, normal, halfway
        )


class GGXNormalDistributionFunction(AbstractNormalDistributionFunction):
    def normal_distribution_function(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the GGX (Trowbridge-Reitz) distribution.

        Args:
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.

        Returns:
            Float[Tensor, "*B 1"]: The GGX distribution function.
        """
        a2 = roughness.square()
        denom = normal_dot_halfway.square() * (a2 - 1) + 1.0

        return a2 / (torch.pi * denom.square()).clip(EPS_DTYPE[denom.dtype])


class GGXNormalDistributionFunctionMediumPrecision(
    AbstractNormalDistributionFunction, abc.ABC
):
    def normal_distribution_function(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the GGX (Trowbridge-Reitz) distribution with medium-precision to avoid gradient issues.
        Taken from https://github.com/google/filament/blob/af079b42a65c46ea33b1eb795a4483e3aeb85323/shaders/src/surface_brdf.fs#L54

        Args:
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            normal (Float[Tensor, "*B 3"]): Normal vector.
            halfway (Float[Tensor, "*B 3"]): Halfway vector.

        Returns:
            Float[Tensor, "*B 1"]: The GGX distribution function.
        """

        normal_cross_halfway = torch.cross(normal, halfway, dim=-1)

        one_minus_normal_dot_halfway_squared = torch.sum(
            normal_cross_halfway.square(), dim=-1, keepdim=True
        )

        a = normal_dot_halfway * roughness
        a2 = a.square()
        k = roughness / (one_minus_normal_dot_halfway_squared + a2)
        d = k.square() * (1.0 / torch.pi)

        def saturate_medium_precision(x):
            medium_precision_max = 65504.0
            medium_precision_min = 0.00006103515625

            return torch.clamp(x, medium_precision_min, medium_precision_max)

        final_result = saturate_medium_precision(d)

        return final_result


class BlinnPhongNormalDistributionFunction(AbstractNormalDistributionFunction):
    def normal_distribution_function(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Blinn-Phong distribution.

        Args:
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.

        Returns:
            Float[Tensor, "*B 1"]: The Blinn-Phong distribution function.
        """
        a2 = roughness.square()
        return 1 / (torch.pi * a2) * normal_dot_halfway.pow(2 / a2 - 2)


class BeckmannNormalDistributionFunction(AbstractNormalDistributionFunction):
    def normal_distribution_function(
        self,
        roughness: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        normal: Float[Tensor, "*B 3"],
        halfway: Float[Tensor, "*B 3"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Beckmann distribution.

        Args:
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.

        Returns:
            Float[Tensor, "*B 1"]: The Beckmann distribution function.
        """
        ndh2 = normal_dot_halfway.square()
        a2 = roughness.square()

        scaler = 1 / (torch.pi * a2 * ndh2.square())
        exp_term = torch.exp((ndh2 - 1) / (a2 * ndh2).clip(EPS_DTYPE[roughness.dtype]))

        return scaler * exp_term


NDF_FUNCTIONS = {
    "ggx": GGXNormalDistributionFunction,
    "ggx_medium_precision": GGXNormalDistributionFunctionMediumPrecision,
    "blinn_phong": BlinnPhongNormalDistributionFunction,
    "beckmann": BeckmannNormalDistributionFunction,
}

# ----------------------------------------------------------------------------
# Geometric Shadowing
# ----------------------------------------------------------------------------


class AbstractGeometricShadowing(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        pass

    def forward(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
        perceptual_roughness: bool = True,
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            perceptual_roughness (bool, optional): If True, the roughness is assumed to be perceptual.
                Otherwise, it is assumed to be linear.

        Returns:
            Float[Tensor, "*B 1"]: The geometric shadowing function.
        """

        if perceptual_roughness:
            roughness = roughness.square()

        return self.geometric_shadowing(
            normal_dot_view,
            normal_dot_light,
            view_dot_halfway,
            normal_dot_halfway,
            roughness,
        )


class ImplicitGeometricShadowing(AbstractGeometricShadowing):
    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the mplicit geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.
            perceptual_roughness (bool, optional): If True, the roughness is assumed to be perceptual.
                Otherwise, it is assumed to be linear.

        Returns:
            Float[Tensor, "*B 1"]: The geometric shadowing function.
        """
        return normal_dot_light * normal_dot_view


class NeumannGeometricShadowing(AbstractGeometricShadowing):
    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Neumann geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector. (Unused)
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector. (Unused)
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough. (Unused)

        Returns:
            Float[Tensor, "*B 1"]: The Neumann geometric shadowing function.
        """
        return (normal_dot_light * normal_dot_view) / (
            torch.max(normal_dot_light, normal_dot_view).clip(
                EPS_DTYPE[normal_dot_light.dtype]
            )
        )


class CookTorranceGeometricShadowing(AbstractGeometricShadowing):
    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Cook-Torrance geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough. (Unused)

        Returns:
            Float[Tensor, "*B 1"]: The geometric shadowing function.
        """
        return torch.min(
            (2 * normal_dot_halfway * normal_dot_view) / view_dot_halfway,
            (2 * normal_dot_halfway * normal_dot_light) / view_dot_halfway,
        ).clip(min=1)


class KelemanGeometricShadowing(AbstractGeometricShadowing):
    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Kelemen geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector. (Unused)
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough. (Unused)

        Returns:
            Float[Tensor, "*B 1"]: The geometric shadowing function.
        """
        return (normal_dot_view * normal_dot_light) / view_dot_halfway.square()


class AbstractSmithGeometricShadowing(AbstractGeometricShadowing, abc.ABC):
    @abc.abstractmethod
    def partial_smith_function(
        self, normal_dot_x: Float[Tensor, "*B 1"], roughness: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the partial Smith function.

        Args:
            normal_dot_x (Float[Tensor, "*B 1"]): Dot product of the normal and x vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The Smith function.
        """
        pass

    def geometric_shadowing(
        self,
        normal_dot_view: Float[Tensor, "*B 1"],
        normal_dot_light: Float[Tensor, "*B 1"],
        view_dot_halfway: Float[Tensor, "*B 1"],
        normal_dot_halfway: Float[Tensor, "*B 1"],
        roughness: Float[Tensor, "*B 1"],
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Smith geometric shadowing function.

        Args:
            normal_dot_view (Float[Tensor, "*B 1"]): Dot product of the normal and view vector.
            normal_dot_light (Float[Tensor, "*B 1"]): Dot product of the normal and light vector.
            view_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the view and halfway vector.
            normal_dot_halfway (Float[Tensor, "*B 1"]): Dot product of the normal and halfway vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The geometric shadowing function.
        """
        return self.partial_smith_function(
            normal_dot_view, roughness
        ) * self.partial_smith_function(normal_dot_light, roughness)


class SmithBeckmannGeometricShadowing(AbstractSmithGeometricShadowing):
    def partial_smith_function(
        self, normal_dot_x: Float[Tensor, "*B 1"], roughness: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Smith-Beckmann geometric function.

        Args:
            normal_dot_x (Float[Tensor, "*B 1"]): Dot product of the normal and view/light vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The geometric function.
        """
        c = normal_dot_x / (roughness * safe_sqrt(1 - normal_dot_x.square()))

        true_term = (3.535 * c + 2.181 * c.square()) / (
            1 + 2.276 * c + 2.577 * c.square()
        )
        false_term = torch.ones_like(c)

        return torch.where(c < 1.6, true_term, false_term)


class SmithGGXGeometricShadowing(AbstractSmithGeometricShadowing):
    def partial_smith_function(
        self, normal_dot_x: Float[Tensor, "*B 1"], roughness: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Smith-GGX geometric function.

        Args:
            normal_dot_x (Float[Tensor, "*B 1"]): Dot product of the normal and view/light vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The geometric function.
        """
        a2 = roughness.square()
        return (2 * normal_dot_x) / (
            normal_dot_x + safe_sqrt(a2 + (1 - a2) * normal_dot_x.square())
        ).clip(EPS_DTYPE[normal_dot_x.dtype])


class SmithSchlickBeckmannGeometricShadowing(AbstractSmithGeometricShadowing):
    def partial_smith_function(
        self, normal_dot_x: Float[Tensor, "*B 1"], roughness: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Smith-Schlick-Beckmann geometric function.

        Args:
            normal_dot_x (Float[Tensor, "*B 1"]): Dot product of the normal and view/light vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The geometric function.
        """
        k = roughness * torch.sqrt(2 / torch.pi)

        return normal_dot_x / (normal_dot_x * (1 - k) + k).clip(
            EPS_DTYPE[normal_dot_x.dtype]
        )


class SmithUE4schlickGGXGeometricShadowing(AbstractSmithGeometricShadowing):
    def partial_smith_function(
        self, normal_dot_x: Float[Tensor, "*B 1"], roughness: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 1"]:
        """
        Computes the Smith-UE4-Schlick-GGX geometric function.

        Args:
            normal_dot_x (Float[Tensor, "*B 1"]): Dot product of the normal and view/light vector.
            roughness (Float[Tensor, "*B 1"]): Roughness of the surface. 0.0 is perfectly smooth, 1.0 is very rough.

        Returns:
            Float[Tensor, "*B 1"]: The geometric function.
        """
        k = roughness / 2

        return normal_dot_x / (normal_dot_x * (1 - k) + k).clip(
            EPS_DTYPE[normal_dot_x.dtype]
        )


GEO_SHADOWING_FUNCTIONS = {
    "implicit": ImplicitGeometricShadowing,
    "neumann": NeumannGeometricShadowing,
    "cook_torrance": CookTorranceGeometricShadowing,
    "kelemen": KelemanGeometricShadowing,
    "smith_beckmann": SmithBeckmannGeometricShadowing,
    "smith_ggx": SmithGGXGeometricShadowing,
    "smith_schlick_beckmann": SmithSchlickBeckmannGeometricShadowing,
    "smith_ue4schlick_ggx": SmithUE4schlickGGXGeometricShadowing,
}

# ----------------------------------------------------------------------------
# Fresnel Term
# ----------------------------------------------------------------------------


class AbstractFresnelTerm(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self, f0: Float[Tensor, "*B 3"], vdh: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 3"]:
        """
        Computes the Fresnel term.

        Args:
            f0 (Float[Tensor, "*B 3"]): The Fresnel reflectance at normal incidence.
            vdh (Float[Tensor, "*B 1"]): The dot product of the view and halfway vectors. (Not used)

        Returns:
            Float[Tensor, "*B 3"]: The Fresnel term.
        """
        pass


class NoFresnelTerm(AbstractFresnelTerm):
    def forward(
        self, f0: Float[Tensor, "*B 3"], vdh: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 3"]:
        """
        Computes the Fresnel term with no Fresnel effect.

        Args:
            f0 (Float[Tensor, "*B 3"]): The Fresnel reflectance at normal incidence.
            vdh (Float[Tensor, "*B 1"]): The dot product of the view and halfway vectors. (Not used)

        Returns:
            Float[Tensor, "*B 3"]: The Fresnel term.
        """
        return f0


class SchlickFresnelTerm(AbstractFresnelTerm):
    def forward(
        self, f0: Float[Tensor, "*B 3"], vdh: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 3"]:
        """
        Computes the Schlick Fresnel term.

        Args:
            f0 (Float[Tensor, "*B 3"]): The Fresnel reflectance at normal incidence.
            vdh (Float[Tensor, "*B 1"]): The dot product of the view and halfway vectors.

        Returns:
            Float[Tensor, "*B 3"]: The Fresnel term.
        """
        return f0 + (1 - f0) * (1 - vdh).pow(5)


class CookTorranceFresnelTerm(AbstractFresnelTerm):
    def forward(
        self, f0: Float[Tensor, "*B 3"], vdh: Float[Tensor, "*B 1"]
    ) -> Float[Tensor, "*B 3"]:
        """
        Computes the Cook-Torrance Fresnel term.

        Args:
            f0 (Float[Tensor, "*B 3"]): The Fresnel reflectance at normal incidence.
            vdh (Float[Tensor, "*B 1"]): The dot product of the view and halfway vectors.

        Returns:
            Float[Tensor, "*B 3"]: The Fresnel term.
        """
        eta = (1 + safe_sqrt(f0)) / (1 - safe_sqrt(f0))
        g = safe_sqrt(eta.square() + vdh.square() - 1)

        return (
            0.5
            * ((g - vdh) / (g + vdh)).square()
            * (1 + (((g + vdh) * vdh - 1) / ((g - vdh) * vdh + 1)).square())
        )


FRESNEL_FUNCTIONS = {
    "no_fresnel": NoFresnelTerm,
    "schlick": SchlickFresnelTerm,
    "cook_torrance": CookTorranceFresnelTerm,
}
