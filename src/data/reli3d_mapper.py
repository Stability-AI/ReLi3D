# ruff: noqa: E402

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sdata.mappers.base import AbstractMapper

from src.constants import Names
from src.image_utils import load_image
from src.utils.color_space import srgb_to_linear
from src.utils.ops import get_intrinsic, get_ray_directions, get_rays
from src.utils.rotation_sixd import matrix_to_rotation_6d
from src.utils.typing import (
    Any,
    Dict,
    Float,
    List,
    Optional,
    Tensor,
    Tuple,
    Union,
)


@dataclass
class ReLi3DMapperConfig:
    num_views_per_scene: int = 0  # If set to 0, autodetects.
    num_views_input: int = 1
    num_views_output: int = 3

    train_input_views: Optional[List[int]] = None
    train_sup_views: str = "random"

    return_first_n_cases: int = -1  # for debugging purpose
    repeat: int = 1  # for debugging purpose

    train_indices: Optional[Tuple[int, int]] = None
    val_indices: Optional[Tuple[int, int]] = None
    test_indices: Optional[Tuple[int, int]] = None

    height: int = 128
    width: int = 128
    rand_min_height: Optional[int] = None
    rand_max_height: int = 128
    rand_max_width: int = 128
    crop_center_in_mask: bool = False

    cond_height: int = 512
    cond_width: int = 512

    batch_size: int = 1
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_input_views: Optional[List[int]] = None
    selected_output_keys: Optional[List[str]] = None

    binarize_mask: bool = True
    binarize_mask_threshold: float = 0.5
    fill_missing_values: bool = True

    use_gt_reni: bool = False

    make_roughness_linear: bool = False
    dataset_is_repaired: bool = False

    rescale_cameras_to_unit: bool = False
    unit_radius: float = 0.5

    # Noise parameters
    add_pose_noise: bool = False
    pose_noise_std_rot: float = (
        0.01  # Standard deviation for rotation noise (e.g., radians for axis-angle)
    )
    pose_noise_std_trans: float = 0.01  # Standard deviation for translation noise

    add_intrinsics_noise: bool = False
    intrinsics_noise_std_focal: float = (
        1.0  # Standard deviation for focal length noise (in pixels)
    )
    intrinsics_noise_std_principal: float = (
        1.0  # Standard deviation for principal point noise (in pixels)
    )


def rescale_cameras_to_unit(
    c2ws: Dict[int, torch.Tensor],
    fov_rads: Dict[int, torch.Tensor],
    unit_radius: float = 0.5,
) -> Tuple[Dict[int, torch.Tensor], float, float]:
    """
    Rescales the translation of every c2w so the object fits in the cube
    [-unit_radius, unit_radius]^3.

    Returns the scale that was applied (use it for depth, point clouds …).
    """
    # 1. compute the largest visible radius among all views
    r_vals = []
    for view_index, c2w in c2ws.items():
        # accept scalar or tensor [fov_x, fov_y]
        fov_scalar = (
            fov_rads[view_index]
            if fov_rads[view_index].numel() == 1
            else torch.max(fov_rads[view_index])
        )
        dist = torch.linalg.norm(c2w[:3, 3])  # ‖t‖
        r_view = dist * torch.sin(0.5 * fov_scalar)  # half-width
        r_vals.append(r_view.item())

    r_max = torch.tensor(r_vals).min()
    if r_max <= unit_radius:
        return c2ws, 1.0, r_max  # identity scale

    # 2. global scale
    scale = unit_radius / r_max

    # 3. apply in-place (clone if you need the originals)
    for view_index, c2w in c2ws.items():
        c2w_new = c2w.clone()
        c2w_new[:3, 3] *= scale
        c2ws[view_index] = c2w_new

    return c2ws, scale, r_max


class ReLi3DMapper(AbstractMapper):
    def __init__(
        self,
        cfg: Any,
        sft_key: str = "safetensors",
        split: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert split in ["train", "val", "test"]
        self.cfg = OmegaConf.merge(OmegaConf.structured(ReLi3DMapperConfig), cfg)

        self.sft_key = sft_key
        self.split = split

    def _get_key(self, sample: Dict, key: str) -> Union[Dict, None]:
        if key not in sample:
            raise KeyError(
                f"Key: {key} not in a current sample with keys {list(sample.keys())}"
            )
        return sample[key]

    def should_add_key(self, key: str) -> bool:
        if self.cfg.selected_output_keys is None:
            return True
        return key in self.cfg.selected_output_keys

    def __call__(self, sample: Dict) -> Union[Dict, None]:
        if not [key for key in sample if key.endswith(self.sft_key)]:
            raise KeyError(
                f"Key: {self.sft_key} not in a current sample with keys {list(sample.keys())}"
            )

        dataset_name = "unknown"
        if "dataset_name" in sample:
            dataset_name = sample["dataset_name"]

        dataset_type = "non-pbr"
        if "dataset_type" in sample:
            dataset_type = sample["dataset_type"]

        is_pbr = dataset_type in ["pbr"]

        # Get sft sample
        sft_id = [key for key in sample if key.endswith(self.sft_key)][0]
        sft_sample = sample.pop(sft_id)

        num_views_per_scene = (
            self.cfg.num_views_per_scene
            if self.cfg.num_views_per_scene > 0
            else len([x for x in sft_sample if x.startswith("fov_rad")])
        )
        assert (
            f"rgb_{num_views_per_scene - 1:04d}" in sft_sample
        ), "Incorrect number of views per scene!"

        if self.cfg.train_input_views is not None:
            assert len(self.cfg.train_input_views) == self.cfg.num_views_input
            cond_ids = self.cfg.train_input_views
        else:
            cond_ids = np.random.choice(
                num_views_per_scene, self.cfg.num_views_input, replace=False
            )

        sup_ids = self._get_sup_ids(cond_ids, num_views_per_scene)

        view_ids = np.concatenate([cond_ids, sup_ids])

        if self.split != "train" and self.cfg.eval_input_views is not None:
            assert len(self.cfg.eval_input_views) == self.cfg.num_views_input
            view_ids[: self.cfg.num_views_input] = self.cfg.eval_input_views

        data_cond, data_sup = [], []

        out = {}

        has_reni = False
        if "reni_latent" in sft_sample:
            has_reni = True
            reni_latent = self._get_key(sft_sample, "reni_latent").view(-1, 3)
            reni_rotation = self._get_key(sft_sample, "reni_rotation").view(3, 3)
            reni_strength = self._get_key(sft_sample, "reni_strength").view(1)
            illumination_z_rotation_rads = self._get_key(
                sft_sample, "illumination_z_rotation_rads"
            ).view(1)
            ilummination_idx = self._get_key(sft_sample, "illumination_idx").view(1)
            reni_rotation_repr = matrix_to_rotation_6d(reni_rotation)

            if self.cfg.use_gt_reni:
                out.update(
                    {
                        Names.RENI_LATENT: reni_latent,
                        Names.ILLUMINATION_ROTATION: reni_rotation,
                        Names.ILLUMINATION_ROTATION.add_suffix(
                            "repr"
                        ): reni_rotation_repr,
                        Names.ILLUMINATION_STRENGTH: reni_strength,
                        Names.ILLUMINATION_Z_ROTATION_RADS: illumination_z_rotation_rads,
                        Names.ILLUMINATION_IDX: ilummination_idx,
                    }
                )
            else:
                out.update(
                    {
                        Names.RENI_LATENT.cond: reni_latent,
                        Names.ILLUMINATION_ROTATION.cond: reni_rotation,
                        Names.ILLUMINATION_ROTATION.add_suffix(
                            "repr"
                        ).cond: reni_rotation_repr,
                        Names.ILLUMINATION_STRENGTH.cond: reni_strength,
                        Names.ILLUMINATION_Z_ROTATION_RADS.cond: illumination_z_rotation_rads,
                        Names.ILLUMINATION_IDX.cond: ilummination_idx,
                    }
                )
        else:
            out.update(
                {
                    Names.RENI_LATENT.cond: torch.ones(49, 3)
                    * Names.RENI_LATENT.empty_indicator_value,
                    Names.ILLUMINATION_ROTATION.cond: torch.ones(3, 3)
                    * Names.ILLUMINATION_ROTATION.empty_indicator_value,
                    Names.ILLUMINATION_ROTATION.add_suffix("repr").cond: torch.ones(6)
                    * Names.ILLUMINATION_ROTATION.empty_indicator_value,
                    Names.ILLUMINATION_STRENGTH.cond: torch.ones(1)
                    * Names.ILLUMINATION_STRENGTH.empty_indicator_value,
                    Names.ILLUMINATION_Z_ROTATION_RADS.cond: torch.ones(1)
                    * Names.ILLUMINATION_Z_ROTATION_RADS.empty_indicator_value,
                    Names.ILLUMINATION_IDX.cond: torch.ones(1)
                    * Names.ILLUMINATION_IDX.empty_indicator_value,
                }
            )

        object_uid = "unknown"
        if "object_uid" in sft_sample:
            object_uid_bytes = self._get_key(sft_sample, "object_uid")
            object_uid = object_uid_bytes.numpy().tobytes().decode("utf-8")

        c2ws = {}
        fov_rads = {}
        for view_index in range(num_views_per_scene):
            if f"fov_rad_{view_index:04d}" in sft_sample:
                c2w = self._get_key(sft_sample, f"c2w_{view_index:04d}")
                fov_rad = self._get_key(sft_sample, f"fov_rad_{view_index:04d}")
                c2ws[view_index] = c2w
                fov_rads[view_index] = fov_rad

        if self.cfg.rescale_cameras_to_unit:
            c2ws, scale, rmax = rescale_cameras_to_unit(
                c2ws, fov_rads, self.cfg.unit_radius
            )
            if scale != 1:
                print(
                    f"Rescaled cameras to unit with scale {scale} from radius {rmax}, dataset is not unit sphere: {dataset_name} {dataset_type} {sft_id} {object_uid}"
                )

        for i, view_index in enumerate(view_ids):
            if i < self.cfg.num_views_input:
                data_cur = data_cond
                crop_height, crop_width = self.cfg.cond_height, self.cfg.cond_width
                resize_height, resize_width = self.cfg.cond_height, self.cfg.cond_width
            else:
                data_cur = data_sup
                if self.split == "train":
                    crop_height, crop_width = self.cfg.height, self.cfg.width
                    resize_height = np.random.randint(
                        self.cfg.rand_min_height or self.cfg.height,
                        self.cfg.rand_max_height + 1,
                    )
                    resize_width = int(
                        np.round(resize_height * self.cfg.width / self.cfg.height)
                    )
                else:
                    crop_height, crop_width = self.cfg.eval_height, self.cfg.eval_width
                    resize_height, resize_width = (
                        self.cfg.eval_height,
                        self.cfg.eval_width,
                    )

            # SFT Dict:
            # {
            #   "fov_rad": array: float, shape=(B)
            #   "c2w": array:float, shape=(B, 4, 4)
            #   "rgb_{view_index:04d}": bytes (jpg)
            #   "reni_latent": array: float, shape=(B, 49, 3)
            #   "reni_rotation": array: float, shape=(B, 3, 3)
            #   "reni_strength": array: float, shape=(B)
            #   "mask_{view_index:04d}": Optional[bytes (jpg)] # Either this or metallicroughmask
            #   "metallicroughmask_{view_index:04d}": Optional[bytes (jpg)] # Either this or mask
            #   "basecolor_{view_index:04d}": Optional[bytes (jpg)] # Can also be diffuse in case of objaverse
            #   "normal_{view_index:04d}": Optional[bytes (exr)]
            #   "depth_{view_index:04d}": Optional[bytes (exr)]
            # }

            rgb_bytes = self._get_key(sft_sample, f"rgb_{view_index:04d}")
            rgb = load_image(
                rgb_bytes,
                is_rgb=True,
                float_output=True,
                resize_size=(resize_width, resize_height),
            )

            # Extrinsic
            c2w = c2ws[view_index]
            # Convert from Blender (Z up, Y forward) to OpenGL (Y up, -Z forward)
            # Rotate 90 degrees around X axis

            if not self.cfg.dataset_is_repaired:
                blender_to_gl = torch.tensor(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                    dtype=c2w.dtype,
                    device=c2w.device,
                )
                c2w = blender_to_gl @ c2w

            # Focal length
            focal_length_xy = torch.zeros(2, device=c2w.device)
            if f"fov_rad_{view_index:04d}" in sft_sample:
                fov_rad = self._get_key(sft_sample, f"fov_rad_{view_index:04d}")
                if not fov_rad.shape:
                    # Calculate initial focal lengths based on FOV and resize dimensions
                    focal_length_x = 0.5 * resize_width / (0.5 * fov_rad).tan()
                    focal_length_y = 0.5 * resize_height / (0.5 * fov_rad).tan()
                    focal_length_xy = torch.tensor(
                        [focal_length_x, focal_length_y], device=c2w.device
                    )
                else:
                    # Assuming fov_rad contains [fov_x, fov_y]
                    focal_length_xy = (
                        0.5
                        * torch.tensor([resize_width, resize_height], device=c2w.device)
                        / (0.5 * fov_rad).tan()
                    )
            else:
                raise ValueError(f"No fov_rad found for view {view_index}")

            # Principal point
            principal_point_xy = torch.tensor(
                [resize_width / 2, resize_height / 2], device=c2w.device
            )
            if f"principal_point_{view_index:04d}" in sft_sample:
                principal_point_xy_orig = self._get_key(
                    sft_sample, f"principal_point_{view_index:04d}"
                )
                org_image = load_image(
                    rgb_bytes,
                    is_rgb=True,
                    float_output=True,
                )  # Get the unscaled original size
                # Apply the resize scaling to the principal point
                scale_x = resize_width / org_image.shape[1]
                scale_y = resize_height / org_image.shape[0]
                principal_point_xy = principal_point_xy_orig * torch.tensor(
                    [scale_x, scale_y], device=c2w.device
                )

            intrinsic = get_intrinsic(focal_length_xy, principal_point_xy)

            # --- Apply Noise (Training and conditioning views Only) ---
            if self.split == "train" and i < self.cfg.num_views_input:
                if self.cfg.add_pose_noise:
                    # Rotation noise (small axis-angle perturbation)
                    axis_angle_noise = (
                        torch.randn(3, device=c2w.device) * self.cfg.pose_noise_std_rot
                    )
                    angle = torch.linalg.norm(axis_angle_noise)
                    if angle > 1e-6:  # Avoid division by zero for near-zero rotation
                        axis = axis_angle_noise / angle
                        # Rodrigues' rotation formula for matrix exponential
                        K = torch.tensor(
                            [
                                [0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0],
                            ],
                            device=c2w.device,
                        )
                        noise_rot_mat = (
                            torch.eye(3, device=c2w.device)
                            + torch.sin(angle) * K
                            + (1 - torch.cos(angle)) * (K @ K)
                        )
                        # Apply rotation noise
                        c2w[:3, :3] = noise_rot_mat @ c2w[:3, :3]

                    # Translation noise
                    noise_trans = (
                        torch.randn(3, device=c2w.device)
                        * self.cfg.pose_noise_std_trans
                    )
                    c2w[:3, 3] = c2w[:3, 3] + noise_trans

                if self.cfg.add_intrinsics_noise:
                    # Add noise to focal lengths
                    noise_focal = (
                        torch.randn(2, device=intrinsic.device)
                        * self.cfg.intrinsics_noise_std_focal
                    )
                    intrinsic[0, 0] = intrinsic[0, 0] + noise_focal[0]
                    intrinsic[1, 1] = intrinsic[1, 1] + noise_focal[1]

                    # Add noise to principal points
                    noise_principal = (
                        torch.randn(2, device=intrinsic.device)
                        * self.cfg.intrinsics_noise_std_principal
                    )
                    intrinsic[0, 2] = intrinsic[0, 2] + noise_principal[0]
                    intrinsic[1, 2] = intrinsic[1, 2] + noise_principal[1]

                    # Update focal_length_xy and principal_point_xy from noisy intrinsics
                    focal_length_xy = torch.tensor(
                        [intrinsic[0, 0], intrinsic[1, 1]], device=intrinsic.device
                    )
                    principal_point_xy = torch.tensor(
                        [intrinsic[0, 2], intrinsic[1, 2]], device=intrinsic.device
                    )

            # --- Recalculate derived values using potentially noisy c2w and intrinsics ---
            directions_unnormed: Float[Tensor, "H W 3"] = get_ray_directions(
                H=resize_height,
                W=resize_width,
                focal=focal_length_xy,  # Use potentially noisy focal length
                principal=principal_point_xy,  # Use potentially noisy principal point
                normalize=False,
            )
            directions: Float[Tensor, "H W 3"] = F.normalize(
                directions_unnormed, dim=-1
            )
            # Use potentially noisy c2w
            rays_o, rays_d_unnormed = get_rays(directions_unnormed, c2w, keepdim=True)
            _, rays_d = get_rays(directions, c2w, keepdim=True)

            # Use potentially noisy c2w
            camera_positions: Float[Tensor, "3"] = c2w[0:3, 3]

            metallic = None
            roughness = None
            basecolor = None
            normal = None
            depth = None
            mask = None
            crop_mask = None

            if f"metallicroughmask_{view_index:04d}" in sft_sample and (
                self.should_add_key("metallic")
                or self.should_add_key("roughness")
                or self.should_add_key("mask")
            ):
                joined_bytes = self._get_key(
                    sft_sample, f"metallicroughmask_{view_index:04d}"
                )
                joined_img = load_image(
                    joined_bytes,
                    is_rgb=True,
                    resize_size=(resize_width, resize_height),
                )

                # FIXME: THIS GOT MESSED UP
                # It is not metallic, roughness, mask
                # It is roughness, metallic, mask
                mask = joined_img[:, :, 2:3]
                if self.cfg.binarize_mask:
                    # Binarize mask with threshold value
                    mask = (mask > self.cfg.binarize_mask_threshold).float()

                roughness = srgb_to_linear(joined_img[:, :, 0:1])
                if self.cfg.make_roughness_linear:
                    roughness = roughness.square()
                roughness = roughness * mask

                metallic = srgb_to_linear(joined_img[:, :, 1:2]) * mask

                # Check if metallic or roughness contains any nan or inf values and are between 0 and 1
                # If any invalid values are found set metallic and roughness to None
                metallic_mask = (
                    (
                        torch.isnan(metallic)
                        | torch.isinf(metallic)
                        | (metallic < 0)
                        | (metallic > 1)
                    )
                    .any()
                    .item()
                )
                if metallic_mask:
                    metallic = None

                roughness_mask = (
                    (
                        torch.isnan(roughness)
                        | torch.isinf(roughness)
                        | (roughness < 0)
                        | (roughness > 1)
                    )
                    .any()
                    .item()
                )
                if roughness_mask:
                    roughness = None

            elif f"nonecropmask_{view_index:04d}" in sft_sample:
                joined_bytes = self._get_key(
                    sft_sample, f"nonecropmask_{view_index:04d}"
                )
                joined_img = load_image(
                    joined_bytes,
                    is_rgb=True,
                    resize_size=(resize_width, resize_height),
                )
                crop_mask = joined_img[:, :, 1:2]
                mask = joined_img[:, :, 2:3]
                if self.cfg.binarize_mask:
                    # Binarize mask with threshold value
                    mask = (mask > self.cfg.binarize_mask_threshold).float()
            elif f"mask_{view_index:04d}" in sft_sample and self.should_add_key("mask"):
                mask_bytes = self._get_key(sft_sample, f"mask_{view_index:04d}")
                mask = load_image(
                    mask_bytes,
                    is_rgb=False,
                    resize_size=(resize_width, resize_height),
                )

                if self.cfg.binarize_mask:
                    # Binarize mask with threshold value
                    mask = (mask > self.cfg.binarize_mask_threshold).float()
            else:
                raise ValueError(
                    f"No mask or metallicroughmask found for view {view_index}"
                )

            if f"basecolor_{view_index:04d}" in sft_sample and self.should_add_key(
                "basecolor"
            ):
                basecolor_bytes = self._get_key(
                    sft_sample, f"basecolor_{view_index:04d}"
                )
                basecolor = load_image(
                    basecolor_bytes,
                    is_rgb=True,
                    resize_size=(resize_width, resize_height),
                )

                # If basecolor contains any nan or inf values set it to None. Also if all values are 0 set it to None
                basecolor_mask = (
                    torch.isnan(basecolor) | torch.isinf(basecolor)
                ).any().item() | (basecolor.max() == 0).item()

                if basecolor_mask:
                    basecolor = None

            if f"normal_{view_index:04d}" in sft_sample and self.should_add_key(
                "normal"
            ):
                normal_bytes = self._get_key(sft_sample, f"normal_{view_index:04d}")
                normal = load_image(
                    normal_bytes,
                    is_rgb=True,
                    resize_size=(resize_width, resize_height),
                )
                normal = normal.clip(min=0, max=1)
                # Normalize
                normal = F.normalize(normal * 2 - 1, dim=-1)

                # Now these are camera space normals. Transfer them to world space.
                # c2w[:3, :3] is the rotation component of camera-to-world transform
                normals_world = normal @ c2w[:3, :3].T
                normal = normals_world * mask

            if f"depth_{view_index:04d}" in sft_sample and self.should_add_key("depth"):
                depth_bytes = self._get_key(sft_sample, f"depth_{view_index:04d}")
                depth = load_image(
                    depth_bytes,
                    is_rgb=False,
                    resize_size=(resize_width, resize_height),
                )

            # Random crop image patches for rendering supervison
            if crop_height < resize_height:
                if self.cfg.crop_center_in_mask or crop_mask is not None:
                    # FIXME: assume square
                    if crop_mask is not None:
                        mask_approx = crop_mask.unsqueeze(0).unsqueeze(0)
                        mask_approx_size = crop_mask.shape[0]
                    else:
                        mask_approx_size = 64
                        mask_approx = F.interpolate(
                            mask.permute(2, 0, 1).unsqueeze(0),
                            (mask_approx_size, mask_approx_size),
                            mode="bilinear",
                            antialias=True,
                        )
                    mask_nonzero = mask_approx.nonzero()
                    if len(mask_nonzero) == 0:
                        # empty render
                        a0 = np.random.randint(0, resize_height - crop_height + 1)
                        b0 = np.random.randint(0, resize_width - crop_width + 1)
                    else:
                        chosen_idx = np.random.choice(range(len(mask_nonzero)))
                        ac = mask_nonzero[chosen_idx][2].item()
                        ac = int(ac * resize_height / mask_approx_size)
                        bc = mask_nonzero[chosen_idx][3].item()
                        bc = int(bc * resize_width / mask_approx_size)
                        ad, be = int(ac - crop_height / 2), int(bc - crop_width / 2)
                        a0 = min(max(0, ad), resize_height - crop_height)
                        b0 = min(max(0, be), resize_width - crop_width)
                else:
                    a0 = np.random.randint(0, resize_height - crop_height + 1)
                    b0 = np.random.randint(0, resize_width - crop_width + 1)
                a1 = a0 + crop_height
                b1 = b0 + crop_width
                rgb = rgb[a0:a1, b0:b1]
                mask = mask[a0:a1, b0:b1]
                if basecolor is not None:
                    basecolor = basecolor[a0:a1, b0:b1]
                if normal is not None:
                    normal = normal[a0:a1, b0:b1]
                if depth is not None:
                    depth = depth[a0:a1, b0:b1]
                if metallic is not None:
                    metallic = metallic[a0:a1, b0:b1]
                if roughness is not None:
                    roughness = roughness[a0:a1, b0:b1]
                rays_o = rays_o[a0:a1, b0:b1]
                rays_d = rays_d[a0:a1, b0:b1]
                rays_d_unnormed = rays_d_unnormed[a0:a1, b0:b1]
                intrinsic[..., 0, 2] -= b0
                intrinsic[..., 1, 2] -= a0

            # Calc normalized intrinsic by crop size (by image height)
            intrinsic_normed = intrinsic.clone()
            intrinsic_normed[..., 0, 2] /= crop_width
            intrinsic_normed[..., 1, 2] /= crop_height
            intrinsic_normed[..., 0, 0] /= crop_width
            intrinsic_normed[..., 1, 1] /= crop_height

            if self.cfg.fill_missing_values:
                if (basecolor is None or not is_pbr) and self.should_add_key(
                    "basecolor"
                ):
                    basecolor = (
                        torch.ones_like(rgb) * Names.BASECOLOR.empty_indicator_value
                    )
                if (normal is None or not has_reni) and self.should_add_key("normal"):
                    normal = (
                        torch.ones_like(rgb)
                        * Names.SHADING_NORMAL.empty_indicator_value
                    )
                # if depth is None and self.should_add_key("depth"):
                #     depth = torch.ones_like(rgb[..., 0:1]) * Names.DEPTH.empty_indicator_value
                if (metallic is None or not is_pbr) and self.should_add_key("metallic"):
                    metallic = (
                        torch.ones_like(rgb[..., 0:1])
                        * Names.METALLIC.empty_indicator_value
                    )
                if (roughness is None or not is_pbr) and self.should_add_key(
                    "roughness"
                ):
                    roughness = (
                        torch.ones_like(rgb[..., 0:1])
                        * Names.ROUGHNESS.empty_indicator_value
                    )

            data_dict = {
                Names.VIEW_INDEX: torch.as_tensor(view_index),
                Names.IMAGE: rgb,
                Names.IMAGE.add_suffix("mask"): rgb * mask,
                Names.IMAGE.add_suffix("bg"): rgb * mask + (1 - mask) * 0.5,
                Names.OPACITY: mask,
                Names.ORIGIN.rays: rays_o,
                Names.DIRECTION.rays: rays_d,
                Names.DIRECTION.rays.add_suffix("unnormed"): rays_d_unnormed,
                Names.CAMERA_POSITION: camera_positions,
                Names.CAMERA_TO_WORLD: c2w,
                Names.INTRINSICS: intrinsic,
                Names.INTRINSICS_NORMED: intrinsic_normed,
            }
            if self.should_add_key("basecolor") and basecolor is not None:
                data_dict[Names.BASECOLOR] = basecolor
            if self.should_add_key("normal") and normal is not None:
                data_dict[Names.SHADING_NORMAL] = normal
            # if self.should_add_key("depth") and depth is not None:
            #     data_dict[Names.DEPTH] = depth
            if self.should_add_key("metallic") and metallic is not None:
                data_dict[Names.METALLIC] = metallic
            if self.should_add_key("roughness") and roughness is not None:
                data_dict[Names.ROUGHNESS] = roughness

            data_cur.append(data_dict)

        data_out = {}
        if data_cond:
            for k in data_cond[0].keys():
                data_out[k.cond] = torch.stack([d[k] for d in data_cond], dim=0)
        if data_sup:
            for k in data_sup[0].keys():
                data_out[k] = torch.stack([d[k] for d in data_sup], dim=0)

        out.update(
            {
                **data_out,
                Names.OBJECT_UID: object_uid,
                Names.DATASET_NAME: dataset_name,
                Names.DATASET_TYPE: dataset_type,
                Names.HEIGHT: torch.as_tensor(crop_height),
                Names.WIDTH: torch.as_tensor(crop_width),
                Names.VIEW_SIZE: torch.as_tensor(self.cfg.num_views_output),
                Names.VIEW_SIZE.cond: torch.as_tensor(self.cfg.num_views_input),
            }
        )

        return out

    def _get_sup_ids(self, cond_ids, num_views_per_scene):
        remain_set = list(set(range(num_views_per_scene)) - set(cond_ids))
        if self.cfg.train_sup_views == "random":
            return np.random.choice(
                num_views_per_scene, self.cfg.num_views_output, replace=False
            )
        elif self.cfg.train_sup_views == "random_remain":
            return np.random.choice(
                remain_set, self.cfg.num_views_output, replace=False
            )
        elif self.cfg.train_sup_views == "random_instant3d":
            return np.random.choice(
                np.concatenate(
                    [
                        cond_ids,
                        np.random.choice(
                            remain_set, self.cfg.num_views_output, replace=False
                        ),
                    ]
                ),
                self.cfg.num_views_output,
                replace=False,
            )
        elif self.cfg.train_sup_views == "random_instant3d_include_first":
            sup_ids = np.random.choice(
                np.concatenate(
                    [
                        cond_ids[1:],
                        np.random.choice(
                            remain_set, self.cfg.num_views_output, replace=False
                        ),
                    ]
                ),
                self.cfg.num_views_output - 1,
                replace=False,
            )
            return np.concatenate([cond_ids[0:1], sup_ids])
        elif self.cfg.train_sup_views == "random_include_input":
            assert self.cfg.num_views_output >= self.cfg.num_views_input
            return np.concatenate(
                [
                    cond_ids,
                    np.random.choice(
                        remain_set,
                        self.cfg.num_views_output - self.cfg.num_views_input,
                        replace=False,
                    ),
                ]
            )
        else:
            raise NotImplementedError
