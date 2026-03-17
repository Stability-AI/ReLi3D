from dataclasses import dataclass, field

import src
from src.constants import Names, OutputsType
from src.models.materials.normal_utils import process_normal
from src.utils.base import BaseModule


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        tone_mapping_cls: str = "src.utils.tonemapping.NoToneMapping"
        tone_mapping: dict = field(default_factory=dict)

        normal_type: str = "geometry"
        radial_up_axis: str = "z"
        slerp_normal_steps: int = 0

    cfg: Config

    def configure(self):
        self.tone_mapping = src.initialize_instance(
            self.cfg.tone_mapping_cls, self.cfg.tone_mapping
        )

    def forward_impl(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError

    @property
    def requires_full_images(self):
        return False

    def forward(
        self, outputs: OutputsType, shade_ray_samples: bool = False
    ) -> OutputsType:
        if shade_ray_samples and self.requires_full_images:
            raise ValueError(
                f"Cannot shade ray samples with {self.__class__.__qualname__}, which requires full images."
            )
        suffix = "ray-samples" if shade_ray_samples else ""

        normal_outputs = {}
        if (
            Names.GEOMETRY_NORMAL.add_suffix(suffix) in outputs
            and Names.SHADING_NORMAL.add_suffix(suffix) not in outputs
        ):
            normal_outputs = process_normal(
                outputs,
                normal_type=self.cfg.normal_type,
                radial_up_axis=self.cfg.radial_up_axis,
                slerp_normal_steps=self.cfg.slerp_normal_steps,
                use_ray_samples=shade_ray_samples,
            )

        evaled = (
            self.forward_impl(outputs | normal_outputs, shade_ray_samples)
            | normal_outputs
        )

        suffixed_radiance = Names.RADIANCE.add_suffix(suffix)
        if suffixed_radiance in evaled:
            evaled[Names.IMAGE.add_suffix(suffix)] = self.tone_mapping(
                evaled[suffixed_radiance]
            )
            evaled[Names.RADIANCE.add_suffix(suffix)] = evaled[suffixed_radiance]
        suffixed_irradiance = Names.IRRADIANCE.add_suffix(suffix)
        if suffixed_irradiance in evaled:
            evaled[Names.TONEMAPPED_IRRADIANCE.add_suffix(suffix)] = self.tone_mapping(
                evaled[suffixed_irradiance]
            )
            evaled[Names.IRRADIANCE.add_suffix(suffix)] = evaled[
                Names.IRRADIANCE.add_suffix(suffix)
            ]

        if Names.SHADING_NORMAL.add_suffix(suffix) in outputs:
            evaled[Names.SHADING_NORMAL.add_suffix(suffix)] = outputs[
                Names.SHADING_NORMAL.add_suffix(suffix)
            ]
        return evaled

    def export_impl(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError

    def export(self, outputs: OutputsType) -> OutputsType:
        out = self.export_impl(outputs)
        out |= process_normal(
            outputs,
            normal_type=self.cfg.normal_type,
            radial_up_axis=self.cfg.radial_up_axis,
            slerp_normal_steps=0,
            use_ray_samples=False,
        )
        return out

    def consumed_keys(self):
        consumed = super().consumed_keys() | {
            Names.GEOMETRY_NORMAL,
            Names.SHADING_NORMAL,
        }
        if not self.requires_full_images:
            consumed |= {
                Names.GEOMETRY_NORMAL.ray_samples,
                Names.SHADING_NORMAL.ray_samples,
            }
        return consumed

    def produced_keys(self):
        produced = super().produced_keys() | {
            Names.IMAGE,
            Names.RADIANCE,
            Names.IRRADIANCE,
            Names.TONEMAPPED_IRRADIANCE,
            Names.VISIBILITY,
            Names.SHADING_NORMAL,
            Names.TANGENT,
            Names.BITANGENT,
        }
        if not self.requires_full_images:
            produced |= {
                Names.IMAGE.ray_samples,
                Names.RADIANCE.ray_samples,
                Names.IRRADIANCE.ray_samples,
                Names.TONEMAPPED_IRRADIANCE.ray_samples,
                Names.VISIBILITY.ray_samples,
                Names.SHADING_NORMAL.ray_samples,
                Names.TANGENT.ray_samples,
                Names.BITANGENT.ray_samples,
            }
        return produced
