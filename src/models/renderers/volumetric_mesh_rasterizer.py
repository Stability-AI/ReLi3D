from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.constants import Names, OutputsType
from src.models.object_representations import AbstractVolumetricRepresentation
from src.models.renderers.base import BaseRenderer
from src.utils.misc import get_device
from src.utils.ops import convert_proj
from src.utils.rasterize import DRTKRasterizerContext, NVDiffRasterizerContext
from src.utils.typing import Float, Tensor

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class MeshRasterizer(BaseRenderer):
    @dataclass
    class Config(BaseRenderer.Config):
        batch_size: int = 1
        near: float = 0.1
        far: float = 1000.0

        context_type: str = "cuda"
        rasterizer: str = "drtk"  # drtk or nvdiffrast

    cfg: Config

    def configure(self) -> None:
        super().configure()
        assert isinstance(self.object_representation, AbstractVolumetricRepresentation)

        if self.cfg.rasterizer == "drtk":
            self.ctx = DRTKRasterizerContext()
        elif self.cfg.rasterizer == "nvdiffrast":
            self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())
        else:
            raise NotImplementedError(f"Unknown rasterizer: {self.cfg.rasterizer}")

    def forward(
        self,
        batch: OutputsType,
    ) -> OutputsType:
        width = batch[Names.WIDTH]
        height = batch[Names.HEIGHT]
        projection_matrix = convert_proj(
            batch[Names.INTRINSICS],
            H=height,
            W=width,
            near=self.cfg.near,
            far=self.cfg.far,
        )
        w2c = (
            batch[Names.WORLD_TO_CAMERA]
            if Names.WORLD_TO_CAMERA in batch
            else torch.linalg.inv(batch[Names.CAMERA_TO_WORLD])
        )
        camera_positions = batch[Names.CAMERA_POSITION]

        batch_size = batch.get(Names.BATCH_SIZE, None)
        n_input = batch.get(Names.VIEW_SIZE, None)

        meshes, queried = self.object_representation.get_mesh(batch)
        if len(meshes) > 0:
            assert (
                len(meshes) == batch_size and batch_size == projection_matrix.shape[0]
            ), f"batch_size: {batch_size}, meshes: {len(meshes)}, projection_matrix: {projection_matrix.shape}"
        n_input = batch.get(Names.VIEW_SIZE, None)
        if n_input is not None:
            assert (
                n_input == projection_matrix.shape[1]
            ), f"n_input: {n_input}, projection_matrix: {projection_matrix.shape}"

        full_out = defaultdict(list)

        empty_mesh_indices = []

        for i, mesh in enumerate(meshes):
            input_dict = {}

            empty_mesh = mesh.v_pos.numel() == 0
            if empty_mesh:
                empty_mesh_indices.append(i)
                continue

            w2c_i = w2c[i] if batch_size is not None else w2c
            projection_matrix_i = (
                projection_matrix[i] if batch_size is not None else projection_matrix
            )
            if n_input is None:
                w2c_i = w2c_i.unsqueeze(0)
                projection_matrix_i = projection_matrix_i.unsqueeze(0)

            with torch.autocast(device_type="cuda", enabled=False):
                v_pos_transformed: Float[Tensor, "BNv 4"] = self.ctx.vertex_transform(
                    mesh.v_pos, w2c_i, projection_matrix_i, (height, width)
                )
                rast, _ = self.ctx.rasterize(
                    v_pos_transformed, mesh.t_pos_idx, (height, width)
                )
                mask = self.ctx.get_mask(rast)
                mask_aa = self.ctx.post_process(
                    mask.float(), rast, v_pos_transformed, mesh.t_pos_idx
                )

                selector = mask[..., 0]

                gb_normal, _ = self.ctx.interpolate_one(
                    mesh.v_nrm, rast, mesh.t_pos_idx
                )
                gb_normal = F.normalize(gb_normal, dim=-1)

                gb_normal_aa = torch.lerp(
                    torch.zeros_like(gb_normal),
                    gb_normal,
                    mask.float(),
                )
                gb_normal_aa = self.ctx.post_process(
                    gb_normal_aa, rast, v_pos_transformed, mesh.t_pos_idx
                )

                gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
                gb_pos_aa = torch.lerp(torch.zeros_like(gb_pos), gb_pos, mask.float())
                gb_pos_aa = self.ctx.post_process(
                    gb_pos_aa, rast, v_pos_transformed, mesh.t_pos_idx
                )
                un_norm_dir = (
                    (
                        camera_positions[i, :, None, None, :]
                        - gb_pos.view(n_input, height, width, 3)
                    )
                    .view(n_input, height, width, 3)
                    .detach()
                )
                gb_viewdirs = F.normalize(un_norm_dir, dim=-1)
                # Transform gb_pos to view space and take the z component
                gb_pos_homo = torch.nn.functional.pad(
                    gb_pos, (0, 1), mode="constant", value=1.0
                )
                gb_depth_map = (
                    torch.matmul(
                        gb_pos_homo.view(n_input, -1, 4), w2c[i].permute(0, 2, 1)
                    )
                    .view(n_input, height, width, 4)
                    .detach()[..., 2:3]
                )

                # Fill in background pixel with high values
                gb_depth_map[~selector] = gb_depth_map.max() * 1.1

            if (nr_active_pixels := selector.sum()) == 0:
                empty_mesh_indices.append(i)
                continue

            input_dict.update(
                {
                    Names.GLOBAL_STEP: batch[Names.GLOBAL_STEP],
                    Names.BATCH_SIZE: n_input,
                    Names.POSITION: gb_pos_aa.view(n_input, height, width, 3),
                    Names.DEPTH: gb_depth_map,
                    Names.OPACITY: mask_aa.view(n_input, height, width, 1),
                    Names.GEOMETRY_NORMAL: gb_normal_aa.view(n_input, height, width, 3),
                    Names.VIEW_DIRECTION: gb_viewdirs,
                    Names.VISIBLE_RAYS: selector,
                    Names.HEIGHT: height,
                    Names.WIDTH: width,
                }
            )

            # `outputs` contains the batched object representation, but we only want to query a single batch element.
            # This dict contains "raysample-like" values for a single batch element, i.e. shaped NxC.
            decoded = self.object_representation(
                {
                    k: v[i]
                    for k, v in batch.items()
                    if k in self.object_representation.consumed_keys()
                },
                gb_pos[selector].view(-1, 3).detach(),
                exclude=self.object_representation.shape_keys(),
            )

            # Some material properties are predicted globally (and are thus present in `batch`), while others
            # are predicted per-pixel and are in `decoded`.
            for material_key in self.material.consumed_keys().intersection(
                {
                    Names.BASECOLOR,
                    Names.DIFFUSE,
                    Names.SPECULAR,
                    Names.ROUGHNESS,
                    Names.METALLIC,
                    Names.SG_AMPLITUDES,
                }
            ):
                if material_key not in decoded and material_key in batch:
                    batch_value = batch[material_key][i]
                    if batch_value.shape[0] == 1:
                        batch_value = batch_value.expand(nr_active_pixels, -1)
                    decoded[material_key] = batch_value

            def dict_restore_images(d):
                def restore_single_image(v):
                    if v.shape[0] != nr_active_pixels:
                        return v
                    full_img = torch.zeros(
                        n_input,
                        height,
                        width,
                        v.shape[-1],
                        device=v.device,
                        dtype=v.dtype,
                    )
                    full_img[selector] = v.squeeze(0)
                    return full_img

                return {k: restore_single_image(v) for k, v in d.items()}

            decoded_images = dict_restore_images(decoded)

            single_out = {
                Names.POSITION: gb_pos_aa.view(n_input, height, width, 3),
                Names.DEPTH: gb_depth_map,
                Names.OPACITY: mask_aa.view(n_input, height, width, 1),
                Names.GEOMETRY_NORMAL: gb_normal_aa.view(n_input, height, width, 3),
            }

            # For now, we only support backgrounds that conform to this signature
            # (i.e. can just create a background image from image specifications).
            single_out.update(
                self.background(
                    {
                        Names.HEIGHT: height,
                        Names.WIDTH: width,
                    }
                )
            )

            single_out.update(decoded_images)
            input_dict.update(single_out)

            # Check what keys are consumed by the material
            consumed_keys = self.material.consumed_keys()
            # Add keys that are not consumed by the material
            for k in consumed_keys:
                if k not in input_dict:
                    if k in batch:
                        input_dict[k] = batch[k][i]

            input_dict[Names.MESH] = mesh

            # FP 16 shading is highly unstable.
            # Only shade full images if the material requires it, otherwise shade per-pixel.
            with torch.autocast(device_type="cuda", enabled=False):
                if self.material.requires_full_images:
                    material_out_images = self.material(
                        input_dict, shade_ray_samples=False
                    )
                else:
                    for key in self.material.consumed_keys():
                        if key not in decoded:
                            # Check if input_dict[key] and selector have the same shape
                            # Except for the last dimensions (if present)
                            if (
                                key in input_dict
                                and input_dict[key].shape[: selector.ndim]
                                == selector.shape
                            ):
                                decoded[key] = input_dict[key][selector].view(
                                    -1, input_dict[key].shape[-1]
                                )
                            if key in batch and key.is_image:
                                print(
                                    f"Expanding {key} from {batch[key][i].shape} to {nr_active_pixels}"
                                )
                                decoded[key] = batch[key][i].expand(
                                    nr_active_pixels, -1
                                )
                    material_out = self.material(
                        {k.ray_samples: v for k, v in decoded.items()}
                        | {
                            Names.GLOBAL_STEP: batch[Names.GLOBAL_STEP],
                            Names.MESH: mesh,
                        },  # FIXME: Yikes...
                        shade_ray_samples=True,
                    )
                    material_out_images = dict_restore_images(
                        {k.ray_accumulated: v for k, v in material_out.items()}
                    )

            # Composite all material outputs with the background.
            for k, v in material_out_images.items():
                # TODO(simon.donne): How to make sure we don't apply background where it doesn't make sense?
                # A good start would be to make sure that `material_out_images` only contains image-like keys.
                # But how much sense is there in applying the background to things like `Names.IRRADIANCE`?
                if isinstance(v, torch.Tensor) and k not in single_out and v.ndim > 3:
                    v_bg = self.background.apply_background(
                        {
                            k: v,
                            Names.BACKGROUND: single_out[Names.BACKGROUND]
                            .unsqueeze(0)
                            .to(v.dtype),
                            Names.OPACITY: single_out[Names.OPACITY],
                        },
                        k,
                    )[k]
                    v_bg_aa = self.ctx.post_process(
                        v_bg, rast, v_pos_transformed, mesh.t_pos_idx
                    )
                    single_out[k] = v_bg_aa.view(n_input, height, width, v.shape[-1])

            for k, v in single_out.items():
                full_out[k].append(v)

        # Support means spoofing relevant keys for the missing images to make sure the batched
        # output still makes sense. However, it's not easy to create meaningful defaults for
        # each of these. And what if all batch elements have invisible meshes? We'd error out
        # during training as none of the parameters would be on the losses' computation graph.
        if len(empty_mesh_indices) == len(meshes):
            return {
                Names.INVALID_BATCH_SENTINEL: True,
            }

        placeholders = {
            k: torch.empty_like(v[0])
            for k, v in full_out.items()
            if isinstance(v[0], torch.Tensor)
        }
        # These come sorted due to `enumerate`, so that in the end everything aligns correctly.
        for i in empty_mesh_indices:
            for k in placeholders:
                full_out[k].insert(i, placeholders[k])

        full_out = {
            k: torch.stack(v, 0) if len(v) > 0 else None
            for k, v in full_out.items()
            if isinstance(v[0], torch.Tensor)
        }
        full_out[Names.MESH] = meshes
        full_out.update(queried)

        return full_out
