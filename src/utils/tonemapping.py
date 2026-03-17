import abc
from dataclasses import dataclass, field
from typing import Tuple

import torch
from jaxtyping import Float

import src

from .asc_cdl import asc_cdl_forward, asc_cdl_reverse
from .color_space import (
    AbstractColorSpaceConversion,
    NoColorSpaceConversion,
    change_luminance,
    linear_rgb_to_luminance,
)
from .config import BaseConfig
from .ops import mix


class AbstractToneMapping(torch.nn.Module, abc.ABC):
    @dataclass
    class Config(BaseConfig):
        color_space_type: str = "src.utils.color_space.NoColorSpaceConversion"
        color_space: dict = field(default_factory=dict)
        exposure: float = 0.0
        gamma: float = 1.0
        enable_cdl: bool = False
        cdl_slope: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
        cdl_offset: Tuple[float, float, float] = field(
            default_factory=lambda: (0.0, 0.0, 0.0)
        )
        cdl_power: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
        cdl_saturation: float = 1.0
        cdl_clamp: bool = False

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._color_space = src.initialize_instance(
            self.cfg.color_space_type, self.cfg.color_space
        )
        self.exposure = 2**self.cfg.exposure
        self.gamma = self.cfg.gamma
        self.enable_cdl = self.cfg.enable_cdl
        self.cdl_slope = torch.nn.Parameter(
            torch.tensor(self.cfg.cdl_slope, dtype=torch.float32), requires_grad=False
        )
        self.cdl_offset = torch.nn.Parameter(
            torch.tensor(self.cfg.cdl_offset, dtype=torch.float32), requires_grad=False
        )
        self.cdl_power = torch.nn.Parameter(
            torch.tensor(self.cfg.cdl_power, dtype=torch.float32), requires_grad=False
        )
        self.cdl_saturation = torch.nn.Parameter(
            torch.tensor(self.cfg.cdl_saturation, dtype=torch.float32),
            requires_grad=False,
        )
        self.cdl_clamp = self.cfg.cdl_clamp
        self.configure()

    def configure(self):
        pass

    @abc.abstractmethod
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass

    def get_value(self, name: str, kwargs: dict):
        if name in kwargs:
            return kwargs[name]
        else:
            return getattr(self, name)

    def forward(
        self, values: Float[torch.Tensor, "*B C"], **kwargs
    ) -> Float[torch.Tensor, "*B C"]:
        pre_tonemapping = values * self.get_value("exposure", kwargs) ** self.get_value(
            "gamma", kwargs
        )
        if self.enable_cdl:
            # Move channel last to first
            pre_tonemapping = pre_tonemapping.permute(0, 3, 1, 2)
            pre_tonemapping = asc_cdl_forward(
                pre_tonemapping,
                self.get_value("cdl_slope", kwargs),
                self.get_value("cdl_offset", kwargs),
                self.get_value("cdl_power", kwargs),
                self.get_value("cdl_saturation", kwargs),
                self.get_value("cdl_clamp", kwargs),
            )
            # Move channel first to last
            pre_tonemapping = pre_tonemapping.permute(0, 2, 3, 1)

        tonemapped = self.forward_impl(pre_tonemapping)

        display_values = self.color_space(tonemapped)

        return display_values

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        raise NotImplementedError("Inverse not implemented")

    def inverse(
        self, values: Float[torch.Tensor, "*B C"], **kwargs
    ) -> Float[torch.Tensor, "*B C"]:
        # First undo color space transform
        linear_values = self.color_space.inverse(values)

        # Undo tonemapping
        untonemapped = self.inverse_impl(linear_values)

        # Undo CDL if enabled
        if self.enable_cdl:
            linear_values = asc_cdl_reverse(
                linear_values,
                self.get_value("cdl_slope", kwargs),
                self.get_value("cdl_offset", kwargs),
                self.get_value("cdl_power", kwargs),
                self.get_value("cdl_saturation", kwargs),
                self.get_value("cdl_clamp", kwargs),
            )

        # Undo pre-tonemapping transform
        return (untonemapped ** (1 / self.get_value("gamma", kwargs))) / self.get_value(
            "exposure", kwargs
        )

    @property
    def supports_inverse(self) -> bool:
        try:
            self.inverse_impl(torch.randn(1, 3))
            return True
        except NotImplementedError:
            return False

    @property
    def color_space(self) -> AbstractColorSpaceConversion:
        return self._color_space

    @property
    def transforms_linear_color_space(self) -> bool:
        return not isinstance(self._color_space, NoColorSpaceConversion)


class NoToneMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values


def inverse_sigmoid(
    values: Float[torch.Tensor, "*B C"], eps: float = 1e-3
) -> Float[torch.Tensor, "*B C"]:
    values = values.clip(min=eps, max=1 - eps)
    values = torch.log(values / (1 - values))
    return values


class SigmoidMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        eps: float = 1e-3

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.sigmoid(values)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return inverse_sigmoid(values, self.cfg.eps)


class LogMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.log(values + 1)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.exp(values) - 1


class SoftClipHDRToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        up_threshold: float = 0.9

    def soft_clip(
        self, values: Float[torch.Tensor, "*B C"], up_threshold: float
    ) -> Float[torch.Tensor, "*B C"]:
        return torch.where(
            torch.less_equal(values, up_threshold),
            values,
            (
                1
                - (1 - up_threshold)
                * torch.exp(-((values - up_threshold) / (1 - up_threshold)))
            ),
        )

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return self.soft_clip(values, self.cfg.up_threshold)


class ReinhardHDRToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        approximate: bool = True

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        if self.cfg.approximate:
            return values / (1 + values)
        else:
            # Short form:
            # L_out = (L_in) / (L_in + 1)
            # C_out = C_in * (L_out / L_in)
            # This becomes:
            # C_out = C_in / (L_in + 1)
            luminance = linear_rgb_to_luminance(values)
            return values / (luminance + 1)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        values = values.clip(0, 1 - 1e-6)
        return -values / (values - 1)


class ReinhardJodieHDRToneMapping(AbstractToneMapping):
    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        luminance = linear_rgb_to_luminance(values)
        tv = values / (1 + values)
        return mix(values / (1 + luminance), tv, tv)


class ExtendedReinhardHDRToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        normalize_first: bool = False
        approximate: bool = True

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        if self.cfg.normalize_first:
            values = (values - values.min()) / (values.max() - values.min())
        max_val = values.max()
        if self.cfg.approximate:
            numerator = values * (1 + (values / max_val.square()))
            return numerator / (1 + values)
        else:
            luminance = linear_rgb_to_luminance(values)
            numerator = luminance * (1 + (luminance / max_val.square()))
            luminance_new = numerator / (1 + luminance)
            return change_luminance(values, luminance_new)


class CheapACESFilmicHDRToneMapping(AbstractToneMapping):
    """This is a cheap ACES Filmic approximation. It is close to the actual but rather expensive ACES Filmic curve.
    Also often used in video games.
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        v = values * 0.6
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return ((v * (a * v + b)) / (v * (c * v + d) + e)).clip(min=0, max=1)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        # Source: wolfram alpha
        values = values.clip(0, 1)
        return -(
            0.833333
            * (
                -3
                + 59 * values
                + torch.sqrt(9 + 13702 * values - 10127 * values.square())
            )
        ) / (-251 + 243 * values)


class CombinedApproximateACESsRGBToneMapping(AbstractToneMapping):
    """This is a combination of the cheap ACES Filmic approximation and a cheap sRGB approximation.
    This is even more approximate compared to the Cheap ACES filmic approximation
    """

    def configure(self):
        if not isinstance(self._color_space, NoColorSpaceConversion):
            raise ValueError(
                "CombinedApproximateACESsRGBToneMapping only supports NoColorSpaceConversion as it handles the sRGB conversion"
            )

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values / (values + 0.1667)

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values / (6 - 6 * values)

    def transforms_linear_color_space(self) -> bool:
        return True


class AgXToneMapping(AbstractToneMapping):
    """AgX tone mapping operator based on Troy Sobotka's implementation.
    Adapted from https://github.com/sobotka/AgX as shown at https://www.shadertoy.com/view/cd3XWr
    """

    @dataclass
    class Config(AbstractToneMapping.Config):
        look_type: str = "default"  # Options: default, golden, punchy
        min_ev: float = -12.47393
        max_ev: float = 4.026069

    def configure(self):
        # AgX input transform matrix
        self.register_buffer(
            "agx_mat",
            torch.tensor(
                [
                    [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                    [0.0784335999999992, 0.878468636469772, 0.0784336],
                    [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
                ]
            ).T,
            persistent=False,
        )

        # AgX inverse matrix
        self.register_buffer(
            "agx_mat_inv",
            torch.tensor(
                [
                    [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                    [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                    [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
                ]
            ).T,
            persistent=False,
        )

        self.min_ev = self.cfg.min_ev
        self.max_ev = self.cfg.max_ev

    def agx_default_contrast_approx(self, x: torch.Tensor) -> torch.Tensor:
        """Logistic function approximation of the AgX contrast curve"""
        A = -0.01
        K = 1.043
        B = 8.1282
        M = 0.6152
        return A + (K - A) / (1.0 + torch.exp(-B * (x - M)))

    def agx_look(self, val: torch.Tensor) -> torch.Tensor:
        """Apply creative look transformation"""
        lw = torch.tensor([0.2126, 0.7152, 0.0722], device=val.device)
        luma = (val * lw.view(1, -1)).sum(dim=-1, keepdim=True)

        # Default parameters with small epsilon to prevent division by zero
        slope = torch.ones_like(val).clamp(min=1e-8)
        power = torch.ones_like(val)
        sat = 1.0

        if self.cfg.look_type == "golden":
            slope = torch.tensor([1.0, 0.9, 0.5], device=val.device).expand_as(val)
            power = torch.full_like(val, 0.8)
            sat = 0.8
        elif self.cfg.look_type == "punchy":
            power = torch.full_like(val, 1.35)
            sat = 1.4

        # Safer power operation by ensuring non-negative inputs
        val = torch.pow(val.clamp(min=0) * slope, power)
        return luma + sat * (val - luma)

    def forward_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        # Input transform
        values = torch.einsum("ij,...j->...i", self.agx_mat, values)

        # More stable log2 with safer clamping
        values = values + 1e-8  # Add small epsilon
        values = torch.log2(values)
        values = values.clamp(min=self.min_ev, max=self.max_ev)
        values = (values - self.min_ev) / (self.max_ev - self.min_ev)

        # Apply contrast and look
        values = self.agx_default_contrast_approx(values)
        values = self.agx_look(values)
        # Output transform
        values = torch.einsum("ij,...j->...i", self.agx_mat_inv, values)
        return values.clamp(min=0, max=1).contiguous() ** 2.2

    def inverse_impl(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        with torch.no_grad():
            # First undo output transform
            values = torch.einsum(
                "ij,...j->...i", self.agx_mat, values.clamp(min=0, max=1) ** (1 / 2.2)
            )

            # Undo look transform
            lw = torch.tensor([0.2126, 0.7152, 0.0722], device=values.device)
            luma = (values * lw.view(1, -1)).sum(dim=-1, keepdim=True)

            # Inverse look parameters
            slope = torch.ones_like(values)
            power = torch.ones_like(values)
            sat = 1.0

            if self.cfg.look_type == "golden":
                slope = torch.tensor([1.0, 0.9, 0.5], device=values.device).expand_as(
                    values
                )
                power = torch.full_like(values, 0.8)
                sat = 0.8
            elif self.cfg.look_type == "punchy":
                power = torch.full_like(values, 1.35)
                sat = 1.4

            # Inverse ASC CDL
            values = luma + (1 / sat) * (values - luma)
            values = torch.pow(values, 1 / power)
            values = values / slope

            # Approximate inverse of the contrast curve using binary search
            # This is more accurate than trying to directly invert the polynomial
            def apply_forward(x):
                return self.agx_default_contrast_approx(x)

            # Binary search for inverse
            left = torch.zeros_like(values)
            right = torch.ones_like(values)
            for _ in range(10):  # Number of binary search iterations
                mid = (left + right) / 2
                mid_mapped = apply_forward(mid)
                left = torch.where(mid_mapped < values, mid, left)
                right = torch.where(mid_mapped >= values, mid, right)
            values = (left + right) / 2

            # Convert back from normalized space to log2 space
            values = values * (self.max_ev - self.min_ev) + self.min_ev
            values = torch.pow(2, values)

            # Undo input transform
            values = torch.einsum("ij,...j->...i", self.agx_mat_inv, values)
            return values.clip(min=0).contiguous()


class LearnedToneMapping(AbstractToneMapping):
    @dataclass
    class Config(AbstractToneMapping.Config):
        epsilon: float = 1e-1
        nr_bins: int = 64
        no_hdr_mapping: bool = False

    def configure(self):
        self.params = torch.nn.Parameter(torch.ones(self.cfg.nr_bins))

    def _transfer_function(self):
        """Returns the learned transfer function, mapping [0, 1] on [0, 1]."""
        # Compute forward CDF from parameters
        partial_hist = torch.nn.functional.softplus(self.params, threshold=5)
        transfer_histogram = torch.cat(
            [torch.zeros_like(partial_hist[:1]), partial_hist]
        ).cumsum(dim=0)
        transfer_histogram = transfer_histogram / transfer_histogram[-1]

        # Define a helper that uses grid_sample for interpolation
        def transfer(x, histogram):
            # scale domain [0,1] to [-1,1] for grid_sample
            return torch.nn.functional.grid_sample(
                histogram.view(1, 1, 1, -1),
                x.view(1, 1, -1, 1).expand(-1, -1, -1, 2) * 2 - 1,
                padding_mode="border",
                align_corners=True,
            ).view(x.shape) * (1 + self.cfg.epsilon)

        return lambda x: transfer(x, transfer_histogram)

    def _inverse_transfer_function(self):
        """Returns the inverse transfer function, mapping [0, 1+epsilon] to [0, 1]."""
        # Reconstruct the CDF from params exactly as in the forward function
        partial_hist = torch.nn.functional.softplus(self.params, threshold=5)
        cdf = torch.cat([torch.zeros_like(partial_hist[:1]), partial_hist]).cumsum(
            dim=0
        )
        cdf = cdf / cdf[-1]

        # Uniform t-grid over [0,1] corresponding to the cdf values
        t = torch.linspace(
            0, 1, steps=self.cfg.nr_bins + 1, device=cdf.device, dtype=cdf.dtype
        )

        # Define the inverse transfer function using piecewise linear interpolation
        def inverse_transfer(y):
            # Scale from [0,1+epsilon] back to [0,1]
            scaled_y = (y / (1 + self.cfg.epsilon)).clamp(0, 1)

            # Use searchsorted to find the bins in which each element of scaled_y lies.
            # searchsorted is not differentiable, but for correctness this is fine.
            idx = torch.searchsorted(cdf, scaled_y, right=True)
            # idx points to the upper bin, so we subtract 1 to get the lower index
            idx = torch.clamp(idx - 1, 0, self.cfg.nr_bins - 1)

            # Gather the cdf and t values at idx and idx+1
            cdf_left = cdf[idx]
            cdf_right = cdf[idx + 1]
            t_left = t[idx]
            t_right = t[idx + 1]

            # Linear interpolation:
            denom = (cdf_right - cdf_left).clamp_min(1e-8)
            frac = (scaled_y - cdf_left) / denom
            x = t_left + frac * (t_right - t_left)

            return x

        return inverse_transfer

    def forward_impl(self, values: torch.Tensor) -> torch.Tensor:
        if self.cfg.no_hdr_mapping:
            mapped = values
        else:
            # Map [0, inf) to [0,1]
            mapped = torch.where(values <= 1, values, (2 - 1 / values)) / 2
        # Apply learned transfer
        transferred = self._transfer_function()(mapped)
        return transferred

    def inverse_impl(self, values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Apply inverse transfer function
            inverse_transferred = self._inverse_transfer_function()(values)

            if self.cfg.no_hdr_mapping:
                return inverse_transferred
            else:
                # Inverse of the forward mapping to [0,1]
                # Original forward: mapped = where(values <= 1, values, (2 - 1/values)) / 2
                # For inverse:
                # If x <= 0.5: values = 2x
                # If x > 0.5: 2x = 2 - 1/values -> values = 1/(2-2x)
                eps = 1e-6
                return torch.where(
                    inverse_transferred <= 0.5,
                    inverse_transferred * 2,
                    1 / (2 - 2 * inverse_transferred.clamp(max=1 - eps)),
                )
