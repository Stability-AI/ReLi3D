from dataclasses import dataclass, field

import torch

import src
from src.constants import FieldName, Names, OutputsType
from src.utils.base import BaseModule
from src.utils.typing import Any, Dict, List, Set


class TransformerPostProcessor(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        preprocessor: List[Dict[str, Any]] = field(default_factory=list)

        tokenizer: List[Dict[str, Any]] = field(default_factory=list)
        input_strategy: str = "concat"  # "concat", "token_concat"
        cross_attention_strategy: str = "concat"  # "concat", "list", "token_concat"

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        postprocessor: List[Dict[str, Any]] = field(default_factory=list)

        output_keys: List[FieldName] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.preprocessor = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.preprocessor[i]["cls"], self.cfg.preprocessor[i]["kwargs"]
                )
                for i in range(len(self.cfg.preprocessor))
            ]
        )

        self.tokenizer = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.tokenizer[i]["cls"], self.cfg.tokenizer[i]["kwargs"]
                )
                for i in range(len(self.cfg.tokenizer))
            ]
        )

        self.postprocessor = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.postprocessor[i]["cls"],
                    self.cfg.postprocessor[i]["kwargs"],
                )
                for i in range(len(self.cfg.postprocessor))
            ]
        )

        self.backbone = src.initialize_instance(
            self.cfg.backbone_cls, self.cfg.backbone
        )

    def forward(self, outputs: OutputsType) -> OutputsType:
        ret = {}
        ret.update(outputs)

        for preprocessor in self.preprocessor:
            ret.update(preprocessor(ret))

        input_conditioning = []
        cross_conditioning = []
        for tokenizer in self.tokenizer:
            if (
                not tokenizer.is_input_tokenizer
                and not tokenizer.is_cross_attention_tokenizer
            ):
                continue
            condition = tokenizer(ret)
            if tokenizer.is_input_tokenizer:
                input_conditioning.append(condition)
            if tokenizer.is_cross_attention_tokenizer:
                cross_conditioning.append(condition)
            ret.update(condition)  # Also directly store whatever we output.

        if len(input_conditioning) > 0:
            input_values = [v for c in input_conditioning for v in c.values()]
            if self.cfg.input_strategy == "concat":
                input_conditioning = torch.cat(input_values, dim=-1)
            elif self.cfg.input_strategy == "token_concat":
                input_conditioning = torch.cat(input_values, dim=-2)
            else:
                raise ValueError(f"Invalid input strategy: {self.cfg.input_strategy}")
            ret[Names.CONDITION.add_suffix("input")] = input_conditioning

        if len(cross_conditioning) > 0:
            cross_values = [v for c in cross_conditioning for v in c.values()]
            # Check if we have 3 and 4 dimensions. If we have 3 we merge all to 3 dimensions by flattening dim 1 to -1
            if any(v.ndim == 3 for v in cross_values):
                cross_values = [
                    v.flatten(1, -2) if v.ndim == 4 else v for v in cross_values
                ]
            if self.cfg.cross_attention_strategy == "concat":
                cross_conditioning = torch.cat(cross_values, dim=-1)
            elif self.cfg.cross_attention_strategy == "token_concat":
                cross_conditioning = torch.cat(cross_values, dim=-2)
            elif self.cfg.cross_attention_strategy == "list":
                cross_conditioning = cross_values
            else:
                raise ValueError(
                    f"Invalid cross attention strategy: {self.cfg.cross_attention_strategy}"
                )
            ret[Names.CONDITION.add_suffix("cross")] = cross_conditioning

        ret.update(
            self.backbone(
                {
                    Names.CONDITION.add_suffix("input"): input_conditioning,
                    Names.CONDITION.add_suffix("cross"): cross_conditioning,
                }
            )
        )

        for tokenizer in self.tokenizer:
            if tokenizer.is_output_tokenizer:
                ret.update(tokenizer.detokenize(ret))

        for postprocessor in self.postprocessor:
            ret.update(postprocessor(ret))

        return {k: ret[k] for k in self.cfg.output_keys}

    def produced_keys(self) -> Set[FieldName]:
        return super().produced_keys().union(set(self.cfg.output_keys))

    def consumed_keys(self) -> Set[FieldName]:
        return (
            super()
            .consumed_keys()
            .union(
                set().union(
                    *[t.consumed_keys() for t in self.preprocessor],
                    *[t.consumed_keys() for t in self.tokenizer],
                    *[t.consumed_keys() for t in self.postprocessor],
                )
            )
        )
