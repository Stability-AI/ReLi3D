from dataclasses import dataclass

from src.constants import Names, OutputsType
from src.models.background.base import BaseBackground
from src.models.materials.base import BaseMaterial
from src.models.object_representations import AbstractObjectRepresentation
from src.utils.base import BaseModule
from src.utils.typing import DictConfig, Optional, Union

try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None


class BaseRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        visibility_threshold: float = 0.5

    cfg: Config

    def __init__(
        self,
        object_representation: AbstractObjectRepresentation,
        material: BaseMaterial,
        background: BaseBackground,
        cfg: Optional[Union[dict, DictConfig]] = None,
    ):
        super().__init__(
            cfg=cfg,
            non_modules={
                "object_representation": object_representation,
                "material": material,
                "background": background,
            },
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError

    def get_textured_mesh(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError

    def consumed_keys(self):
        # We assume that the object representation correctly communicates the consumed keys.
        return super().consumed_keys() | {
            Names.INTRINSICS,
            Names.CAMERA_TO_WORLD,
            Names.WORLD_TO_CAMERA,
            Names.CAMERA_POSITION,
            Names.VIEW_SIZE,
        }.union(
            self.material.consumed_keys(),
            self.background.consumed_keys(),
            self.object_representation.consumed_keys(),
        )

    def produced_keys(self):
        return (
            super()
            .produced_keys()
            .union(
                {
                    Names.POSITION,
                    Names.DEPTH,
                    Names.OPACITY,
                    Names.GEOMETRY_NORMAL,
                    Names.VIEW_DIRECTION,
                    Names.VISIBLE_RAYS,
                    Names.MESH,
                }
            )
            .union(
                self.material.produced_keys(),
                self.background.produced_keys(),
                self.object_representation.produced_keys(),
            )
        )

    @property
    def object_representation(self) -> AbstractObjectRepresentation:
        return self.non_module("object_representation")

    @property
    def material(self) -> BaseMaterial:
        return self.non_module("material")

    @property
    def background(self) -> BaseBackground:
        return self.non_module("background")
