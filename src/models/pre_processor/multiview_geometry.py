from dataclasses import dataclass

import torch
import torchvision

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.ops import get_plucker_rays, normalize, reflect
from src.utils.rotation_sixd import matrix_to_rotation_6d, rotation_6d_to_matrix


class ReflectedViewDirections(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        img_scale: bool = False  # If True, rescales [-1, 1] -> [0, 1]

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        camera_ray_directions = outputs[
            Names.DIRECTION.cond.rays
        ]  # From the camera to the surface.
        surface_normals = outputs[Names.SHADING_NORMAL.cond]
        reflected_view_directions = reflect(
            camera_ray_directions, surface_normals
        )  # From the surface towards the environment.
        if self.cfg.img_scale:
            reflected_view_directions = (reflected_view_directions + 1) / 2
        return {
            Names.REFLECTED_VIEW_DIRECTION.cond: reflected_view_directions,
        }


class ShadingNormalFromDepth(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        img_scale: bool = False  # If True, rescales [-1, 1] -> [0, 1]
        in_key: FieldName = Names.IMAGE.cond
        out_key: FieldName = Names.SHADING_NORMAL.cond

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        depth = outputs[self.cfg.in_key]
        origins = outputs[Names.ORIGIN.rays.cond]
        directions = outputs[Names.DIRECTION.rays.add_suffix("unnormed").cond]
        world_positions = origins + depth * directions
        # Horizontal diff yields one tangent, vertical diff yields another.
        # the cross product yields the normals, in world space.
        mask = outputs[Names.OPACITY.cond]
        d_horz_1 = (
            world_positions[:, :, 1:-1, 1:-1] - world_positions[:, :, 1:-1, :-2]
        ) * mask[:, :, 1:-1, :-2]
        d_horz_2 = (
            world_positions[:, :, 1:-1, 2:] - world_positions[:, :, 1:-1, 1:-1]
        ) * mask[:, :, 1:-1, 2:]
        d_horz = normalize(d_horz_1 + d_horz_2)
        d_vert_1 = (
            world_positions[:, :, 1:-1, 1:-1] - world_positions[:, :, :-2, 1:-1]
        ) * mask[:, :, 1:-1, :-2]
        d_vert_2 = (
            world_positions[:, :, 2:, 1:-1] - world_positions[:, :, 1:-1, 1:-1]
        ) * mask[:, :, 2:, 1:-1]
        d_vert = normalize(d_vert_1 + d_vert_2)
        geometry_normals_world = torch.zeros_like(origins)
        geometry_normals_world[:, :, 1:-1, 1:-1] = torch.cross(d_horz, d_vert, dim=-1)
        shading_normals = (geometry_normals_world).clip(min=-1, max=1)
        if self.cfg.img_scale:
            shading_normals = (shading_normals + 1) / 2

        return {self.cfg.out_key: shading_normals}


class ShadingNormalFromRGB(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        img_scale: bool = True  # If True, rescales [-1, 1] -> [0, 1]
        model_name: str = "metric3d_vit_large"

        in_key: FieldName = Names.IMAGE.cond
        out_key: FieldName = Names.SHADING_NORMAL.cond

    cfg: Config

    def configure(self):
        self.model = torch.hub.load(
            "yvanyin/metric3d", self.cfg.model_name, pretrain=True
        )
        self.register_buffer(
            "mean",
            torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None] / 255.0,
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None] / 255.0,
            persistent=False,
        )

    def prepare_input(self, rgb):
        rgb_prep = (
            rgb.view(-1, *rgb.shape[-3:]).permute(0, 3, 1, 2) - self.mean[None]
        ) / self.std[None]

        # keep ratio resize
        target_size = (616, 1064)  # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = rgb_prep.shape[-2:]
        scale = min(target_size[0] / h, target_size[1] / w)
        rgb_prep = torchvision.transforms.functional.resize(
            rgb_prep, size=(int(h * scale), int(w * scale)), antialias=True
        )

        h, w = rgb_prep.shape[-2:]
        pad_h = target_size[0] - h
        pad_w = target_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_prep = torchvision.transforms.functional.pad(
            rgb_prep,
            padding=(pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half),
            fill=0,
            padding_mode="constant",
        )
        padding_info = (pad_h_half, pad_w_half, pad_h - pad_h_half, pad_w - pad_w_half)

        return rgb_prep, padding_info

    def process_output(self, img, padding_info, target_size):
        # unpad
        if padding_info[0] > 0:
            img = img[:, :, padding_info[0] :, :]
        if padding_info[1] > 0:
            img = img[:, :, :, padding_info[1] :]
        if padding_info[2] > 0:
            img = img[:, :, : -padding_info[2], :]
        if padding_info[3] > 0:
            img = img[:, :, :, : -padding_info[3]]
        # resize
        img = torchvision.transforms.functional.resize(
            img, size=target_size, antialias=True
        )
        # Transform into our default camera coordinate system.
        img = torch.stack([img[:, 0, :, :], -img[:, 1, :, :], -img[:, 2, :, :]], dim=-1)
        return img

    def forward(self, outputs: OutputsType) -> OutputsType:
        rgb = outputs[self.cfg.in_key]
        rgb_prep, padding_info = self.prepare_input(rgb)
        pred_depth, confidence, output_dict = self.model.inference({"input": rgb_prep})
        shading_normals = self.process_output(
            output_dict["prediction_normal"][:, :3, :, :],
            padding_info,
            rgb.shape[-3:-1],
        ).view(rgb.shape)

        if self.cfg.img_scale:
            shading_normals = (shading_normals + 1) / 2

        return {self.cfg.out_key: shading_normals}


class PluckerRays(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        img_scale: bool = False  # If True, rescales [-1, 1] -> [0, 1]

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        camera_ray_directions = outputs[
            Names.DIRECTION.cond.rays
        ]  # From the camera to the surface.
        origins = outputs[Names.ORIGIN.cond.rays]
        plucker_rays = get_plucker_rays(origins, camera_ray_directions)
        if self.cfg.img_scale:
            plucker_rays = (plucker_rays + 1) / 2
        return {
            Names.PLUCKER_RAYS.cond: plucker_rays,
        }

    def produced_keys(self):
        return super().produced_keys().union({Names.PLUCKER_RAYS.cond})

    def consumed_keys(self):
        return (
            super()
            .consumed_keys()
            .union({Names.DIRECTION.cond.rays, Names.ORIGIN.cond.rays})
        )


class RotationMatrixToRepresentation(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_key: FieldName = Names.ILLUMINATION_ROTATION
        out_key: FieldName = Names.ILLUMINATION_ROTATION.add_suffix("repr")

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        matrices = outputs[self.cfg.in_key]
        return {
            self.cfg.out_key: matrix_to_rotation_6d(matrices),
        }


class RepresentationToRotationMatrix(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_key: FieldName = Names.ILLUMINATION_ROTATION.add_suffix("repr")
        out_key: FieldName = Names.ILLUMINATION_ROTATION

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        representations = outputs[self.cfg.in_key]
        return {
            self.cfg.out_key: rotation_6d_to_matrix(representations),
        }

    def produced_keys(self):
        return super().produced_keys().union({self.cfg.out_key})

    def consumed_keys(self):
        return super().consumed_keys().union({self.cfg.in_key})
