import abc
from dataclasses import dataclass

import trimesh

from src.constants import FieldName, OutputsType
from src.models.materials.base import BaseMaterial
from src.models.mesh import Mesh
from src.utils.base import BaseModule
from src.utils.typing import List, Optional, Set, Tuple


class AbstractObjectRepresentation(BaseModule, abc.ABC):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    @abc.abstractmethod
    def forward(self, outputs: OutputsType) -> OutputsType:
        pass

    @abc.abstractmethod
    def get_mesh(self, outputs: OutputsType) -> Tuple[List[Mesh], OutputsType]:
        pass

    @abc.abstractmethod
    def get_textured_trimesh(
        self,
        outputs: OutputsType,
        material: BaseMaterial,
        texture_resolution: int = 1024,
        remesh: Optional[str] = None,
        vertex_count: int = -1,
    ) -> List[trimesh.Trimesh]:
        pass

    def consumed_keys(self) -> Set[FieldName]:
        return set()
