import abc
from dataclasses import dataclass

from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.config import BaseConfig
from src.utils.typing import Optional


class AbstractTokenizer(BaseModule, abc.ABC):
    @dataclass
    class Config(BaseConfig):
        tokenize_key: FieldName = Names.CONDITION
        is_input_tokenizer: bool = False
        is_cross_attention_tokenizer: bool = False
        is_output_tokenizer: bool = False
        detokenize_key: Optional[FieldName] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if self.cfg.detokenize_key is None:
            self.cfg.detokenize_key = self.cfg.tokenize_key

    @abc.abstractmethod
    def forward(self, outputs: OutputsType) -> OutputsType:
        pass

    def detokenize(self, outputs: OutputsType) -> OutputsType:
        raise NotImplementedError(
            f"Detokenization not implemented for {self.__class__.__name__}"
        )

    @property
    def is_input_tokenizer(self) -> bool:
        return self.cfg.is_input_tokenizer

    @property
    def is_cross_attention_tokenizer(self) -> bool:
        return self.cfg.is_cross_attention_tokenizer

    @property
    def is_output_tokenizer(self) -> bool:
        return self.cfg.is_output_tokenizer
