import os
from dataclasses import dataclass

import torch.nn as nn
from safetensors.torch import load_model

from src.constants import FieldName
from src.utils.config import BaseConfig
from src.utils.misc import get_device, load_module_weights
from src.utils.typing import Any, DictConfig, Optional, Set, Union


class Updateable:
    def do_update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except RuntimeError:
                continue
            except AttributeError:
                continue  # ignore attributes like property, which can't be retrieved using getattr
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except RuntimeError:
                continue
            except AttributeError:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


def update_end_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step_end(epoch, global_step)


class BaseObject(Updateable):
    @dataclass
    class Config(BaseConfig):
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig, Config]] = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # if cfg is a dict, convert it to a Config
        if isinstance(cfg, dict):
            cfg = self.Config(**cfg)
        elif isinstance(cfg, DictConfig):
            cfg = self.Config(**cfg)

        self.cfg = cfg
        self.device = get_device()
        self.configure()

    def configure(self) -> None:
        pass


class BaseModule(nn.Module, Updateable):
    @dataclass
    class Config(BaseConfig):
        weights: Optional[str] = None

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self,
        cfg: Optional[Union[dict, DictConfig, Config]] = None,
        non_modules: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # if cfg is a dict, convert it to a Config
        if isinstance(cfg, dict):
            cfg = self.Config(**cfg)
        elif isinstance(cfg, DictConfig):
            cfg = self.Config(**cfg)

        self.cfg = cfg
        self.device = get_device()
        self._non_modules = non_modules or {}
        self.configure()
        if hasattr(self.cfg, "weights") and self.cfg.weights is not None:
            # format: path/to/weights:module_name
            vals = self.cfg.weights.split(":")
            # Check if the weights_path ends with safetensors
            if os.path.splitext(vals[0])[1] == ".safetensors":
                print(f"Loading weights from {vals[0]}")
                if not vals[0].startswith("/"):
                    vals[0] = os.path.join(
                        os.path.dirname(__file__), "..", "..", vals[0]
                    )
                load_model(self, vals[0])
            else:
                weights_path, module_name = vals
                if not weights_path.startswith("/"):
                    weights_path = os.path.join(
                        os.path.dirname(__file__), "..", "..", weights_path
                    )
                print(f"Loading weights from {weights_path}:{module_name}")
                state_dict, epoch, global_step = load_module_weights(
                    weights_path, module_name=module_name, map_location="cpu"
                )
                self.load_state_dict(state_dict)
                self.do_update_step(
                    epoch, global_step, on_load_weights=True
                )  # restore states

        self.post_configure()

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        pass

    def register_non_module(self, name: str, module: nn.Module) -> None:
        # non-modules won't be treated as model parameters
        self._non_modules[name] = module

    def non_module(self, name: str):
        return self._non_modules.get(name, None)

    inplace_module = False  # Set this to true to remove consumption/production checks for a given module class.

    def consumed_keys(self) -> Set[FieldName]:
        """Returns the _local_ keys consumed by this class.

        This excludes global keys like `Names.GLOBAL_STEP` or `Names.WIDTH`
        that can be expected to be present at all times.
        It is not a hard requirement to pass all of the keys in, but it
        should be a guaranteed superset of consumed keys, i.e. the classes
        should never consume something which isn't listed.
        """
        return set()

    def produced_keys(self) -> Set[FieldName]:
        """Returns the keys produced by this material class.

        It is not a hard requirement to produce all of the keys, but it
        should be a guaranteed superset of consumed keys, i.e. the classes
        should never produce something which isn't listed.
        """
        return set()

    def get_basemodule_children(self, filter_inplace: bool = True):
        children = {}
        new_children = {k: v for (k, v) in self.named_children()}
        while new_children:
            newer_children = {}
            for child_path, child in new_children.items():
                newer_children |= {
                    child_path + "." + k: v for (k, v) in child.named_children()
                }
            children |= new_children
            new_children = {
                k: v for k, v in newer_children.items() if k not in children
            }
        children = {
            k: v
            for k, v in children.items()
            if isinstance(v, BaseModule)
            and (not v.inplace_module or not filter_inplace)
        }  # Inplace modules are exempt from these checks, as they do not affect key sets.
        return children
