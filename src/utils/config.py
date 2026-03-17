import dataclasses
import os
from dataclasses import dataclass, fields, is_dataclass

from omegaconf import OmegaConf

import src
from src.constants import FieldName, Names
from src.utils.typing import Any, DictConfig, List, Optional, Type, Union

# ============ Register OmegaConf Resolvers ============= #
if not OmegaConf.has_resolver("calc_exp_lr_decay_rate"):
    OmegaConf.register_new_resolver(
        "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
    )

if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver("add", lambda a, b: a + b)
if not OmegaConf.has_resolver("sub"):
    OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
if not OmegaConf.has_resolver("div"):
    OmegaConf.register_new_resolver("div", lambda a, b: a / b)
if not OmegaConf.has_resolver("idiv"):
    OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
if not OmegaConf.has_resolver("basename"):
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
if not OmegaConf.has_resolver("rmspace"):
    OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
if not OmegaConf.has_resolver("tuple2"):
    OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
if not OmegaConf.has_resolver("gt0"):
    OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
if not OmegaConf.has_resolver("not"):
    OmegaConf.register_new_resolver("not", lambda s: not s)

# ======================================================= #


@dataclass
class BaseConfig:
    def __post_init__(self):
        for data_field in fields(self):
            # Cast strings to FieldName when typed as FieldName or Optional[FieldName]
            if (data_field.type is FieldName) or (
                hasattr(data_field.type, "_name")
                and data_field.type._name == "Optional"
                and (data_field.type.__args__[0] is FieldName)
            ):
                if isinstance(value := getattr(self, data_field.name), str):
                    setattr(self, data_field.name, Names(value))
            if (
                hasattr(data_field.type, "_name")
                and data_field.type._name == "List"
                and (data_field.type.__args__[0] is FieldName)
                and len(value := getattr(self, data_field.name)) > 0
                and isinstance(value[0], str)
            ):
                setattr(self, data_field.name, [Names(v) for v in value])

            # Recursively instantiate nested dataclass configs
            if is_dataclass(data_field.type):
                if isinstance(value := getattr(self, data_field.name), dict):
                    setattr(self, data_field.name, data_field.type(**value))

            if (
                hasattr(data_field.type, "_name")
                and data_field.type._name == "List"
                and is_dataclass(data_field.type.__args__[0])
            ):
                if isinstance(
                    value := getattr(self, data_field.name), list
                ) and isinstance(value[0], dict):
                    setattr(
                        self,
                        data_field.name,
                        [data_field.type.__args__[0](**v) for v in value],
                    )


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def instantiate_config(
    config_class: Type[dataclasses.dataclass],
    cfg: Optional[Union[dict, DictConfig]] = None,
) -> dataclasses.dataclass:
    """Instantiate a config dataclass from a dictionary/OmegaConf object."""
    dataclass_fields = dataclasses.fields(config_class)
    field_names = {f.name for f in dataclass_fields}

    cfg = cfg or {}
    relevant_cfg = {k: v for k, v in cfg.items() if k in field_names}
    data = config_class(**relevant_cfg)

    if extra_cfg_names := set(cfg.keys()) - field_names:
        src.warn(
            "Ignoring keys [" + ", ".join(extra_cfg_names) + "] as they are not supported by " + config_class.__qualname__
        )
    if missing_cfg_names := set(
        field.name
        for field in dataclass_fields
        if getattr(data, field.name) is dataclasses.MISSING
    ):
        src.warn(
            "Missing keys [" + ", ".join(missing_cfg_names) + "] for " + config_class.__qualname__
        )

    return data
