from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import trimesh

import src
from src.constants import Names, OutputsType
from src.utils.typing import Any, Dict, List, Optional, Tuple

from .abstract_system import AbstractSystem

if TYPE_CHECKING:
    from src.main_module import MainModule


class FeedForwardSystem(AbstractSystem):
    @dataclass
    class Config(AbstractSystem.Config):
        preprocessor: List[Dict[str, Any]] = field(default_factory=list)

        tokenizer: List[Dict[str, Any]] = field(default_factory=list)
        input_strategy: str = "concat"  # "concat", "token_concat"
        cross_attention_strategy: str = "concat"  # "concat", "list", "token_concat"

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        postprocessor: List[Dict[str, Any]] = field(default_factory=list)

        material_cls: str = ""
        material: dict = field(default_factory=dict)

        background_cls: str = ""
        background: dict = field(default_factory=dict)

        object_representation_cls: str = ""
        object_representation: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        self.preprocessor = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.preprocessor[i]["cls"],
                    self.cfg.preprocessor[i].get("kwargs", {}),
                )
                for i in range(len(self.cfg.preprocessor))
            ]
        )

        self.tokenizer = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.tokenizer[i]["cls"],
                    self.cfg.tokenizer[i].get("kwargs", {}),
                )
                for i in range(len(self.cfg.tokenizer))
            ]
        )

        self.postprocessor = torch.nn.ModuleList(
            [
                src.initialize_instance(
                    self.cfg.postprocessor[i]["cls"],
                    self.cfg.postprocessor[i].get("kwargs", {}),
                )
                for i in range(len(self.cfg.postprocessor))
            ]
        )

        self.background = src.initialize_instance(
            self.cfg.background_cls, self.cfg.background
        )
        self.backbone = src.initialize_instance(
            self.cfg.backbone_cls, self.cfg.backbone
        )
        self.material = src.initialize_instance(
            self.cfg.material_cls, self.cfg.material
        )
        self.object_representation = src.initialize_instance(
            self.cfg.object_representation_cls, self.cfg.object_representation
        )

        self.renderer = src.initialize_instance(
            self.cfg.renderer_cls,
            cfg=self.cfg.renderer,
            object_representation=self.object_representation,
            material=self.material,
            background=self.background,
        )

    def forward(
        self,
        batch: OutputsType,
        global_step: int,
        skip_rendering: bool = False,  # FIXME: I do not like this flag here.
    ) -> OutputsType:
        ret = {}
        ret.update(batch)
        ret[Names.GLOBAL_STEP] = global_step

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
            if self.cfg.input_strategy == "concat":
                input_conditioning = torch.cat(
                    [v for c in input_conditioning for v in c.values()], dim=-1
                )
            elif self.cfg.input_strategy == "token_concat":
                input_conditioning = torch.cat(
                    [v for c in input_conditioning for v in c.values()], dim=-2
                )
            else:
                raise ValueError(f"Invalid input strategy: {self.cfg.input_strategy}")
            ret[Names.CONDITION.add_suffix("input")] = input_conditioning

        if len(cross_conditioning) > 0:
            if self.cfg.cross_attention_strategy == "concat":
                cross_conditioning = torch.cat(
                    [v for c in cross_conditioning for v in c.values()], dim=-1
                )
            elif self.cfg.cross_attention_strategy == "token_concat":
                cross_conditioning = torch.cat(
                    [v for c in cross_conditioning for v in c.values()], dim=-2
                )
            elif self.cfg.cross_attention_strategy == "list":
                cross_conditioning = [v for c in cross_conditioning for v in c.values()]
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

        if not skip_rendering:
            ret.update(self.renderer(ret))

        return ret

    # FIXME: This is a hack to get the mesh.
    def get_mesh(
        self,
        batch: OutputsType,
        texture_resolution: int = 1024,
        remesh: Optional[str] = None,
        vertex_count: int = -1,
    ) -> Tuple[List[trimesh.Trimesh], OutputsType]:
        outputs = self.forward(batch, 0, skip_rendering=True)
        return self.object_representation.get_textured_trimesh(
            outputs, self.material, texture_resolution, remesh, vertex_count
        ), outputs

    def log_viz(
        self,
        main_module: "MainModule",
        prefix: str,
        gt: OutputsType,
        pred: OutputsType,
        max_images: int = 16,
    ):
        trimeshes = self.object_representation.get_textured_trimesh(pred, self.material)
        for idx, mesh in enumerate(trimeshes):
            if idx >= max_images:
                break
            main_module.save_mesh(
                f"it{main_module.true_global_step}-{prefix}-mesh{idx}.glb",
                mesh,
                name=f"{prefix}_viz/mesh",
                step=main_module.true_global_step,
            )
