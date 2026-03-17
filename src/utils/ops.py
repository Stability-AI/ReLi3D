import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function

import src
from src.utils.typing import (
    Any,
    Callable,
    Dict,
    Float,
    List,
    Num,
    Optional,
    Tensor,
    Tuple,
    Union,
)


def face_forward(normal: Float[Tensor, "*B 3"], vec: Float[Tensor, "*B 3"]):
    return torch.where(dot(normal, vec) < 0, -normal, normal)


def dot(x, y, dim=-1):
    return torch.sum(x * y, dim, keepdim=True)


def reflect(x, n):
    return x - 2 * dot(x, n) * n


def mix(a, b, t):
    return a * (1 - t) + b * t


def slerp(a, b, t, require_normalize: bool = False, dim: int = -1):
    if require_normalize:
        a = normalize(a, dim=dim)
        b = normalize(b, dim=dim)

    omega = torch.acos(dot(a, b).clip(-1, 1))
    so = torch.sin(omega)
    # avoid division by zero
    so_fix = torch.where(so.abs() < 1e-6, 1e-6, so)
    return torch.where(
        so.abs() < 1e-6,
        (1.0 - t) * a + t * b,
        torch.sin((1.0 - t) * omega) / so_fix * a + torch.sin(t * omega) / so_fix * b,
    )


EPS_DTYPE = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
    torch.float32: 1e-7,
    torch.float64: 1e-8,
}


def safe_sqrt(x, eps=None):
    if eps is None:
        eps = EPS_DTYPE[x.dtype]
    return x.clip(min=eps).sqrt()


def safe_exp(x, max=15):
    return torch.exp(torch.clamp(x, max=max))


def safe_acos(x, eps=None):
    if eps is None:
        eps = EPS_DTYPE[x.dtype]
    return torch.acos(torch.clamp(x, -1 + eps, 1 - eps))


def smoothstep(x, edge0=0.0, edge1=1.0):
    x = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


class _SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / x.clamp(min=torch.finfo(x.dtype).eps)


def safe_log(x):
    return _SafeLog.apply(x)


def normalize(x, dim=-1, eps=None):
    if eps is None:
        eps = EPS_DTYPE[x.dtype]
    return F.normalize(x, dim=dim, p=2, eps=eps)


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none" or name == "linear" or name == "identity":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif "sigmoid" in name:
        if "/" in name:
            _act, min, max = name.split("/")
            min = float(min)
            max = float(max)
            return lambda x: torch.sigmoid(x) * (max - min) + min
        else:
            return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "negative":
        return lambda x: -x
    elif name == "normalize_channel_last":
        return lambda x: normalize(x)
    elif name == "normalize_channel_first":
        return lambda x: normalize(x, dim=1)
    elif name == "2_channel_normal_to_3_channel_normal":

        def _2_channel_normal_to_3_channel_normal(x):
            z = safe_sqrt(1.0 - x[..., 0:1] ** 2 - x[..., 1:2] ** 2)
            return normalize(torch.cat([x, z], dim=-1))

        return _2_channel_normal_to_3_channel_normal
    elif "override" in name:
        if "/" in name:
            override_name, override_value = name.split("/")
            override_value = float(override_value)
            return lambda x: x * 0.0 + override_value
        else:
            raise ValueError(
                f"Invalid override value must be in the format of 'override/value', got {name}"
            )
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        print(x.shape, x.dtype)
        return self.lambd(x)


class BiasLayer(torch.nn.Module):
    def __init__(self, bias: Union[float, List[float]]):
        super().__init__()
        self.register_buffer(
            "bias", torch.tensor(bias, dtype=torch.float32), persistent=False
        )

    def forward(self, x):
        return x + self.bias.to(x.dtype)


class MultiplierLayer(torch.nn.Module):
    def __init__(self, multiplier: Union[float, List[float]]):
        super().__init__()
        self.register_buffer(
            "multiplier",
            torch.tensor(multiplier, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x):
        return x * self.multiplier.to(x.dtype)


class ActivationModule(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x):
        return get_activation(self.name)(x)

    def __repr__(self):
        return f"ActivationModule({self.name})"


def get_activation_module(name) -> torch.nn.Module:
    return ActivationModule(name)


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float) or len(focal.shape) == 0 or len(focal) == 1:
        fx, fy = focal, focal
    else:
        fx, fy = focal

    if principal is not None:
        cx, cy = principal
    else:
        cx, cy = W / 2, H / 2

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    if normalize:
        directions = F.normalize(directions, dim=-1)

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=False,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_intrinsic_from_fov(fov, H, W, bs=-1):
    if isinstance(fov, float) or len(fov.shape) == 0 or len(fov) == 1:
        focal_length_x = 0.5 * H / np.tan(0.5 * fov)
        focal_length_y = 0.5 * W / np.tan(0.5 * fov)
    else:
        focal_length_x = 0.5 * H / np.tan(0.5 * fov[0])
        focal_length_y = 0.5 * W / np.tan(0.5 * fov[1])
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length_x
    intrinsic[1, 1] = focal_length_y
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)


def get_intrinsic(focal_length_xy, principal_point_xy, bs=-1):
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length_xy[0]
    intrinsic[1, 1] = focal_length_xy[1]
    intrinsic[0, 2] = principal_point_xy[0]
    intrinsic[1, 2] = principal_point_xy[1]

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        src.warn("Empty rays_indices!")
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([0]).to(ray_indices)
        t_end = torch.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end


def rays_intersect_bbox(
    rays_o: Float[Tensor, "N 3"],
    rays_d: Float[Tensor, "N 3"],
    radius: Float,
    near: Float = 0.0,
    valid_thresh: Float = 0.01,
    background: bool = False,
):
    input_shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    rays_d_valid = torch.where(
        rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d
    )
    if type(radius) in [int, float]:
        radius = torch.FloatTensor(
            [[-radius, radius], [-radius, radius], [-radius, radius]]
        ).to(rays_o.device)
    radius = (
        (1.0 - 1.0e-3) * radius
    )  # tighten the radius to make sure the intersection point lies in the bounding box
    interx0 = (radius[..., 1] - rays_o) / rays_d_valid
    interx1 = (radius[..., 0] - rays_o) / rays_d_valid
    t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(interx0, interx1).amin(dim=-1)

    # check wheter a ray intersects the bbox or not
    rays_valid = t_far - t_near > valid_thresh

    # t_near_valid, t_far_valid = t_near[rays_valid], t_far[rays_valid]
    # global_near = t_near_valid.min().item()
    # global_far = t_far_valid.max().item()

    t_near[torch.where(~rays_valid)] = 0.0
    t_far[torch.where(~rays_valid)] = 0.0

    t_near = t_near.view(*input_shape, 1)
    t_far = t_far.view(*input_shape, 1)
    rays_valid = rays_valid.view(*input_shape)

    return t_near, t_far, rays_valid


def get_plucker_rays(
    rays_o: Float[Tensor, "*N 3"], rays_d: Float[Tensor, "*N 3"]
) -> Float[Tensor, "*N 6"]:
    rays_o = F.normalize(rays_o, dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    return torch.cat([rays_o.cross(rays_d, dim=-1), rays_d], dim=-1)


def c2w_to_polar(c2w: Float[Tensor, "4 4"]) -> Tuple[float, float, float]:
    cam_pos = c2w[:3, 3]
    x, y, z = cam_pos.tolist()
    distance = cam_pos.norm().item()
    elevation = math.asin(z / distance)
    if abs(x) < 1.0e-5 and abs(y) < 1.0e-5:
        azimuth = 0
    else:
        azimuth = math.atan2(y, x)
        if azimuth < 0:
            azimuth += 2 * math.pi

    return elevation, azimuth, distance


def euler_angles_to_rot_matrix(
    x: float, y: float, z: float, order: str = "XYZ"
) -> Float[Tensor, "3 3"]:
    """
    Convert Blender rotation order XYZ to rotation matrix.
    """
    if len(set(order)) != 3:
        raise ValueError("Invalid rotation order.")
    cx, cy, cz = torch.cos(torch.tensor([x, y, z], dtype=torch.float32)).unbind()
    sx, sy, sz = torch.sin(torch.tensor([x, y, z], dtype=torch.float32)).unbind()

    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)

    matrix = {
        "x": rx,
        "y": ry,
        "z": rz,
    }

    order = order.lower()
    return matrix[order[0]] @ matrix[order[1]] @ matrix[order[2]]


def polar_to_c2w(
    elevation: float, azimuth: float, distance: float
) -> Float[Tensor, "4 4"]:
    """
    Compute L = p - C.
    Normalize L.
    Compute s = L x u. (cross product)
    Normalize s.
    Compute u' = s x L.
    rotation = [s, u, -l]
    """
    z = distance * math.sin(elevation)
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)

    # Construct a look-at matrix
    return create_lookat_c2w(x, y, z)


def create_lookat_c2w(x: float, y: float, z: float) -> Float[Tensor, "4 4"]:
    lv = -torch.as_tensor([x, y, z]).float()
    lv = F.normalize(lv, dim=0)

    # Up vec
    uv = torch.as_tensor([0.0, 0.0, 1.0]).float()
    # Right vec
    sv = lv.cross(uv, dim=-1)
    sv = F.normalize(sv, dim=0)
    # Recompute up vec
    uv = sv.cross(lv, dim=-1)
    uv = F.normalize(uv, dim=0)

    rot = torch.stack([sv, uv, -lv], dim=0).T
    c2w = torch.zeros((4, 4), dtype=torch.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = torch.as_tensor([x, y, z])
    c2w[3, 3] = 1
    return c2w


def convert3x4_4x4(mat: Float[Tensor, "B 3 4"]) -> Float[Tensor, "B 4 4"]:
    addition = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=mat.device).view(
        1, 1, 4
    ) * torch.ones_like(mat[:, 0:1])

    return torch.cat([mat, addition], 1)  # (N, 4, 4)


def stable_invert_rotation_translation_matrix(
    matrix: Float[Tensor, "*B 4 4"],
) -> Float[Tensor, "*B 4 4"]:
    r = matrix[..., :3, :3].view(-1, 3, 3)
    t = matrix[..., :3, 3].view(-1, 3)

    r_inv = torch.transpose(r, 1, 2)
    t_inv = torch.matmul(r_inv, t.unsqueeze(-1)).squeeze(-1)

    invert_RT = torch.concat([r_inv, -t_inv[..., None]], -1)
    inv_RT_4x4 = convert3x4_4x4(invert_RT)

    return inv_RT_4x4.view(matrix.shape)


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = stable_invert_rotation_translation_matrix(c2w)
    # w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    # w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    # w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    # w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def get_full_projection_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    return (c2w.unsqueeze(0).bmm(proj_mtx.unsqueeze(0))).squeeze(0)


def convert_proj(K: Float[Tensor, "*B 3 3"], H: int, W: int, near: float, far: float):
    o = torch.zeros_like(K[..., 0, 0])
    i = torch.ones_like(K[..., 0, 0])
    far = torch.full_like(K[..., 0, 0], far)
    near = torch.full_like(K[..., 0, 0], near)
    return torch.stack(
        [
            torch.stack(
                [
                    2 * K[..., 0, 0] / W,
                    -2 * K[..., 0, 1] / W,
                    (W - 2 * K[..., 0, 2]) / W,
                    o,
                ],
                dim=-1,
            ),
            torch.stack(
                [o, -2 * K[..., 1, 1] / H, (H - 2 * K[..., 1, 2]) / H, o], dim=-1
            ),
            torch.stack(
                [o, o, (-far - near) / (far - near), -2 * far * near / (far - near)],
                dim=-1,
            ),
            torch.stack([o, o, -i, o], dim=-1),
        ],
        -2,
    )

    # proj = torch.zeros(*K.shape[:-2], 4, 4, dtype=torch.float32, device=K.device)
    # proj[..., 0, 0] = 2 * K[..., 0, 0] / W
    # proj[..., 0, 1] = -2 * K[..., 0, 1] / W
    # proj[..., 0, 2] = (W - 2 * K[..., 0, 2]) / W

    # proj[..., 1, 1] = -2 * K[..., 1, 1] / H
    # proj[..., 1, 2] = (H - 2 * K[..., 1, 2]) / H

    # proj[..., 2, 2] = (-far - near) / (far - near)
    # proj[..., 2, 3] = -2 * far * near / (far - near)

    # proj[..., 3, 2] = -1

    # return proj


def normalize_pc_bbox(pc, scale=1.0):
    # get the bounding box of the mesh
    assert len(pc.shape) in [2, 3] and pc.shape[-1] in [3, 6, 9]
    n_dim = len(pc.shape)
    device = pc.device
    pc = pc.cpu()
    if n_dim == 2:
        pc = pc.unsqueeze(0)
    normalize_pc = []
    for b in range(pc.shape[0]):
        xyz = pc[b, :, :3]  # [N, 3]
        bound_x = (xyz[:, 0].max(), xyz[:, 0].min())
        bound_y = (xyz[:, 1].max(), xyz[:, 1].min())
        bound_z = (xyz[:, 2].max(), xyz[:, 2].min())
        # get the center of the bounding box
        center = np.array(
            [
                (bound_x[0] + bound_x[1]) / 2,
                (bound_y[0] + bound_y[1]) / 2,
                (bound_z[0] + bound_z[1]) / 2,
            ]
        )
        # get the largest dimension of the bounding box
        scale = max(
            bound_x[0] - bound_x[1], bound_y[0] - bound_y[1], bound_z[0] - bound_z[1]
        )
        xyz = (xyz - center) / scale
        extra = pc[b, :, 3:]
        normalize_pc.append(torch.cat([xyz, extra], dim=-1))
    return (
        torch.stack(normalize_pc, dim=0).to(device)
        if n_dim == 3
        else normalize_pc[0].to(device)
    )


def spherical_to_cartesian(
    theta: Float[Tensor, "*B 1"], phi: Float[Tensor, "*B 1"]
) -> Float[Tensor, "*B 3"]:
    """Converts theta and phi (physical notation) to a cartesian unit vector.
    The cartesian coordinates are defined with z up, x right, y forward.
    Phi 0 indicates right, pi/2 forward, pi left, pi/2*3 back
    Theta 0 indicates up and pi down

    Args:
        theta (Tensor[..., 1]): The elevation 0 to pi
        phi (Tensor[..., 1]): The azimuth 0 to 2pi

    Returns:
        Tensor[..., 3]: Cartesian coordinate direction with unit length
    """
    sinTheta = theta.sin()
    cosTheta = theta.cos()
    sinPhi = phi.sin()
    cosPhi = phi.cos()

    return torch.cat((sinTheta * cosPhi, sinTheta * sinPhi, cosTheta), -1)


def cartesian_to_spherical(
    v: Float[Tensor, "*B 3"],
) -> Tuple[Float[Tensor, "*B 1"], Float[Tensor, "*B 1"]]:
    """Converts a cartesian unit vector into theta and phi (physical notation)
    The cartesian coordinates are defined with z up, x right, y forward.
    Phi 0 indicates right, pi/2 forward, pi left, pi/2*3 back
    Theta 0 indicates up and pi down

    Args:
        v (Tensor[..., 3]): A unit length cartesian vector

    Returns:
        Tuple[Tensor[..., 1], Tensor[..., 1]]: A tuple containing
            theta (Tensor[..., 1]): The elevation 0 to pi
            phi (Tensor[..., 1]): The azimuth 0 to 2pi
    """
    vx, vy, vz = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    theta = safe_acos(vz)
    phi = torch.atan2(vy, vx)

    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)

    return (theta, phi)


def coordinate_system(
    a: Float[Tensor, "*B 3"],
) -> Tuple[Float[Tensor, "*B 3"], Float[Tensor, "*B 3"]]:
    """Finds a suitable tangent and bi-tangent to construct a coordinate
    system. As this is impossible the resulting coordinate system can only
    be used for isotropic materials.
    """
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]

    invLenTrue = 1.0 / safe_sqrt(ax * ax + az * az)
    invLenFalse = 1.0 / safe_sqrt(ay * ay + az * az)

    c = normalize(
        torch.where(
            (ax.abs() > ay.abs()).unsqueeze(-1).repeat(*[1 for _ in ax.shape], 3),
            torch.stack((az * invLenTrue, torch.zeros_like(ax), -ax * invLenTrue), -1),
            torch.stack(
                (torch.zeros_like(ax), az * invLenFalse, -ay * invLenFalse), -1
            ),
        )
    )

    b = normalize(torch.cross(c, a, dim=-1))

    return (b, c)
