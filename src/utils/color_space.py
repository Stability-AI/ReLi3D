import abc
from dataclasses import dataclass

import torch
from jaxtyping import Float


def srgb_to_linear(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.04045
    return torch.where(
        torch.greater(x, switch_val),
        ((x.clip(min=switch_val) + 0.055) / 1.055).pow(2.4),
        x / 12.92,
    )


def linear_to_srgb(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.0031308
    return torch.where(
        torch.greater(x, switch_val),
        1.055 * x.clip(min=switch_val).pow(1.0 / 2.4) - 0.055,
        x * 12.92,
    )


def linear_to_rec2020(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    alpha = 1.09929682680944
    beta = 0.018053968510807

    return torch.where(
        torch.greater(x, beta),
        alpha * x.clip(min=1e-8).pow(0.45) - (alpha - 1),
        x * 4.5,
    )


def rec2020_to_linear(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    alpha = 1.09929682680944
    beta = 0.018053968510807

    switch_val = beta * 4.5

    return (
        torch.where(
            torch.greater(x, switch_val),
            (x.clip(min=-(alpha - 1)) + (alpha - 1)) / alpha,
            x / 4.5,
        )
        .clip(min=1e-8)
        .pow(1.0 / 0.45)
    )  # important: clip and pow after


# ------------------------------------------------------------------
# ARRI LogC3  (EI 800, exposure values, V3)
# ------------------------------------------------------------------
_LOGC3_PARAMS = {
    "cut": 0.010591,
    "a": 5.555556,
    "b": 0.052272,
    "c": 0.247190,
    "d": 0.385537,
    "e": 5.367655,  # slope of the toe
    "f": 0.092809,  # intercept so that both branches meet
}

_CUT = _LOGC3_PARAMS["cut"]
_A = _LOGC3_PARAMS["a"]
_B = _LOGC3_PARAMS["b"]
_C = _LOGC3_PARAMS["c"]
_D = _LOGC3_PARAMS["d"]
_E = _LOGC3_PARAMS["e"]
_F = _LOGC3_PARAMS["f"]


def arri_logc_transfer(x: torch.Tensor) -> torch.Tensor:
    """
    ARRI LogC3 **encoding** (linear → LogC value) for EI 800 exposure values.

    Args:
        x: scene-linear exposure value, normalised so that 18 % grey is 0.18.

    Returns:
        LogC-encoded value in the range 0 – 1.
    """
    x = torch.clamp(x, min=0.0)
    log_branch = _C * torch.log10(_A * x + _B) + _D
    lin_branch = _E * x + _F
    return torch.where(x > _CUT, log_branch, lin_branch)


def arri_logc_inverse_transfer(y: torch.Tensor) -> torch.Tensor:
    """
    ARRI LogC3 **decoding** (LogC value → linear) for EI 800 exposure values.

    Args:
        y: LogC-encoded value (0 – 1).

    Returns:
        Scene-linear exposure value (0 – 1, but can exceed 1 for highlights).
    """
    y = torch.clamp(y, min=0.0)
    lin_branch = (y - _F) / _E
    log_branch = (10 ** ((y - _D) / _C) - _B) / _A
    return torch.where(y > _E * _CUT + _F, log_branch, lin_branch)


def xyz_to_cielab(xyz: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    """Convert from XYZ (linear) to CIELAB."""
    # Assume D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # normalized D65 white

    x, y, z = xyz.unbind(-1)
    x = x / Xn
    y = y / Yn
    z = z / Zn

    epsilon = 0.008856
    kappa = 903.3

    def f(t):
        return torch.where(
            t > epsilon, t.clip(min=epsilon).pow(1 / 3), (kappa * t + 16) / 116
        )

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return torch.stack([L, a, b], dim=-1)


def cielab_to_xyz(lab: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    """Convert from CIELAB to XYZ (linear)."""
    L, a, b = lab.unbind(-1)

    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    epsilon = 0.008856
    kappa = 903.3

    def f_inv(t):
        t3 = t.clip(min=0) ** 3  # optional super-safe
        return torch.where(t3 > epsilon, t3, (116 * t - 16) / kappa)

    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)

    # Assume D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    X = xr * Xn
    Y = yr * Yn
    Z = zr * Zn

    return torch.stack([X, Y, Z], dim=-1)


def gamma_to_linear(
    x: Float[torch.Tensor, "*B C"], gamma: float = 2.2
) -> Float[torch.Tensor, "*B C"]:
    return x.pow(gamma)


def linear_to_gamma(
    x: Float[torch.Tensor, "*B C"], gamma: float = 2.2
) -> Float[torch.Tensor, "*B C"]:
    return x.pow(1.0 / gamma)


def linear_rgb_to_luminance(
    rgb: Float[torch.Tensor, "*batch 3"],
) -> Float[torch.Tensor, "*batch 1"]:
    r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def change_luminance(
    rgb: Float[torch.Tensor, "*batch 3"],
    new_luminance: Float[torch.Tensor, "*batch 1"],
) -> Float[torch.Tensor, "*batch 3"]:
    return rgb * new_luminance / linear_rgb_to_luminance(rgb)


def rgb_to_hsl(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin

    # Avoid division by zero
    delta_safe = torch.where(delta == 0, torch.ones_like(delta), delta)

    # Compute h
    h0 = ((rgb[:, 1:2] - rgb[:, 2:3]) / delta_safe) % 6
    h1 = ((rgb[:, 2:3] - rgb[:, 0:1]) / delta_safe) + 2
    h2 = ((rgb[:, 0:1] - rgb[:, 1:2]) / delta_safe) + 4
    h = torch.where(delta == 0, torch.zeros_like(delta), torch.zeros_like(delta))
    h = torch.where((cmax_idx == 0) & (delta != 0), h0, h)
    h = torch.where((cmax_idx == 1) & (delta != 0), h1, h)
    h = torch.where((cmax_idx == 2) & (delta != 0), h2, h)
    h = h / 6.0

    # Compute l
    lum = (cmax + cmin) / 2.0

    # Compute s
    s = torch.zeros_like(l)
    mask = delta != 0
    s_lte_05 = (lum <= 0.5) & mask
    s_gt_05 = (lum > 0.5) & mask
    s = torch.where(s_lte_05, (delta / (lum * 2.0)), s)
    s = torch.where(s_gt_05, (delta / (-lum * 2.0 + 2.0)), s)

    return torch.cat([h, s, l], dim=1)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin

    # Avoid division by zero
    delta_safe = torch.where(delta == 0, torch.ones_like(delta), delta)

    # Compute h
    h0 = ((rgb[:, 1:2] - rgb[:, 2:3]) / delta_safe) % 6
    h1 = ((rgb[:, 2:3] - rgb[:, 0:1]) / delta_safe) + 2
    h2 = ((rgb[:, 0:1] - rgb[:, 1:2]) / delta_safe) + 4
    h = torch.where(delta == 0, torch.zeros_like(delta), torch.zeros_like(delta))
    h = torch.where((cmax_idx == 0) & (delta != 0), h0, h)
    h = torch.where((cmax_idx == 1) & (delta != 0), h1, h)
    h = torch.where((cmax_idx == 2) & (delta != 0), h2, h)
    h = h / 6.0

    # Compute s
    s = torch.where(cmax == 0, torch.zeros_like(delta), delta / cmax.clip(min=1e-6))

    # v
    v = cmax

    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c

    h6 = (h * 6).floor()
    zeros = torch.zeros_like(h)
    conds = [
        (h6 == 0),
        (h6 == 1),
        (h6 == 2),
        (h6 == 3),
        (h6 == 4),
        (h6 == 5),
    ]
    rgb_candidates = [
        torch.cat([c, x, zeros], dim=1),
        torch.cat([x, c, zeros], dim=1),
        torch.cat([zeros, c, x], dim=1),
        torch.cat([zeros, x, c], dim=1),
        torch.cat([x, zeros, c], dim=1),
        torch.cat([c, zeros, x], dim=1),
    ]
    rgb = torch.zeros_like(hsv)
    for cond, cand in zip(conds, rgb_candidates):
        rgb = torch.where(cond, cand, rgb)
    rgb = rgb + m
    return rgb


def hsl_to_rgb(hsl: torch.Tensor) -> torch.Tensor:
    h, s, lum = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    c = (1 - torch.abs(2 * lum - 1)) * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = lum - c / 2

    h6 = (h * 6).floor()
    zeros = torch.zeros_like(h)
    conds = [
        (h6 == 0),
        (h6 == 1),
        (h6 == 2),
        (h6 == 3),
        (h6 == 4),
        (h6 == 5),
    ]
    rgb_candidates = [
        torch.cat([c, x, zeros], dim=1),
        torch.cat([x, c, zeros], dim=1),
        torch.cat([zeros, c, x], dim=1),
        torch.cat([zeros, x, c], dim=1),
        torch.cat([x, zeros, c], dim=1),
        torch.cat([c, zeros, x], dim=1),
    ]
    rgb = torch.zeros_like(hsl)
    for cond, cand in zip(conds, rgb_candidates):
        rgb = torch.where(cond, cand, rgb)
    rgb = rgb + m
    return rgb


class AbstractColorSpaceConversion(torch.nn.Module, abc.ABC):
    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.configure()

    def configure(self):
        pass

    @abc.abstractmethod
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass

    @abc.abstractmethod
    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        pass


class NoColorSpaceConversion(AbstractColorSpaceConversion):
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return values


class LinearToSRGBColorSpaceConversion(AbstractColorSpaceConversion):
    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return linear_to_srgb(values)

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return srgb_to_linear(values)


class LinearToGammaColorSpaceConversion(AbstractColorSpaceConversion):
    @dataclass
    class Config(AbstractColorSpaceConversion.Config):
        gamma: float = 2.2

    def forward(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return linear_to_gamma(values, self.cfg.gamma)

    def inverse(
        self, values: Float[torch.Tensor, "*B C"]
    ) -> Float[torch.Tensor, "*B C"]:
        return gamma_to_linear(values, self.cfg.gamma)


def pq_to_linear(x: torch.Tensor, L_p: float = 10000.0) -> torch.Tensor:
    m1, m2 = 2610 / 16384, 2523 / 32
    c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
    x = torch.clamp(x, 0.0, 1.0) ** (1.0 / m2)
    return (
        torch.pow(torch.maximum(x - c1, torch.zeros_like(x)) / (c2 - c3 * x), 1.0 / m1)
        * L_p
    )


def linear_to_pq(lin: torch.Tensor, L_p: float = 10000.0) -> torch.Tensor:
    m1, m2 = 2610 / 16384, 2523 / 32
    c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
    y = torch.clamp(lin / L_p, 0.0, 1.0) ** m1
    return torch.pow((c1 + c2 * y) / (1 + c3 * y), m2)


_HLG_A = 0.17883277
_HLG_B = 0.28466892  # 1 − 4 a
_HLG_C = 0.55991073  # 0.5 − a ln(4 a)
_HLG_BREAK_L = 1.0 / 12.0  # Linear-light breakpoint
_HLG_BREAK_V = 0.5  # Code-value breakpoint


def linear_to_hlg(lin: torch.Tensor) -> torch.Tensor:
    """
    Convert **scene-referred linear light** (0 – 1) to HLG non-linear signal.

    Args:
        lin: Tensor in any shape whose values are in [0 , 1].

    Returns:
        Tensor of the same shape containing HLG-encoded values in [0 , 1].
    """
    lin = torch.clamp(lin, 0.0, 1.0)
    below = lin <= _HLG_BREAK_L
    v1 = torch.sqrt(3.0 * lin)
    v2 = _HLG_A * torch.log(12.0 * lin - _HLG_B) + _HLG_C
    return torch.where(below, v1, v2)


def hlg_to_linear(v: torch.Tensor) -> torch.Tensor:
    """
    Convert HLG signal values (0 – 1) back to **scene-referred linear light**.

    Args:
        v: Tensor in any shape whose values are in [0 , 1].

    Returns:
        Tensor of the same shape with linear values in [0 , 1].
    """
    v = torch.clamp(v, 0.0, 1.0)
    below = v <= _HLG_BREAK_V
    l1 = (v * v) / 3.0
    l2 = (torch.exp((v - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    return torch.where(below, l1, l2)
