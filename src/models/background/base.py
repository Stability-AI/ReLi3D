from dataclasses import dataclass

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError

    def apply_background(
        self, outputs: OutputsType, key: FieldName, mask_key: FieldName = Names.OPACITY
    ) -> OutputsType:
        if Names.BACKGROUND not in outputs:
            outputs.update(self(outputs))
        img = outputs[key]
        mask = outputs[mask_key]
        bg = outputs[Names.BACKGROUND]
        if img.shape[-1] == 1:
            bg = bg.mean(dim=-1, keepdim=True)
        img = (img * mask + bg * (1 - mask)).to(img.dtype)
        return {key: img, Names.BACKGROUND: bg}

    def produced_keys(self):
        return super().produced_keys().union({Names.BACKGROUND})
