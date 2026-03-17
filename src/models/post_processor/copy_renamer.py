from dataclasses import dataclass, field

from src.constants import FieldName, OutputsType
from src.utils.base import BaseModule


class CopyRenamer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        key_in: FieldName = field(default_factory=FieldName)
        key_out: FieldName = field(default_factory=FieldName)

    cfg: Config

    def forward(self, outputs: OutputsType) -> OutputsType:
        if self.cfg.key_in not in outputs:
            return {}
        return {self.cfg.key_out: outputs[self.cfg.key_in]}
