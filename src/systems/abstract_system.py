import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    # Inference-only runtime does not require Lightning.
    def rank_zero_only(fn):
        return fn

import src
from src.constants import OutputsType
from src.utils.base import BaseModule
from src.utils.misc import HierarchicallyTrackedDict

if TYPE_CHECKING:
    from src.main_module import MainModule


class AbstractSystem(BaseModule, abc.ABC):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_key_contract_checked = False

    @abc.abstractmethod
    def forward(self, batch: OutputsType, step: int) -> OutputsType:
        pass

    @rank_zero_only
    def dkcc_wrap_forwards(self):
        relevant_basemodules = self.get_basemodule_children(filter_inplace=True)

        def named_forward_hooks(module_name):
            def pre_forward_hook(module, args):
                # Make sure args[0] is a dict-like object.
                if not args:
                    return
                args0 = args[0]
                if not isinstance(args0, dict):
                    # raise ValueError(f"BaseModules should take a dict-like object as input.")
                    src.warn(
                        f"{module_name} ({module.__class__}) does not take a dict-like object as input."
                    )
                elif isinstance(args0, HierarchicallyTrackedDict):
                    # If something is already tracking this, then we only wish to start a new level of tracking (but afterwards pass on everything that was accessed to our parents).
                    args0.push_hierarchy()
                else:
                    # Otherwise, we need to start tracking.
                    args0 = HierarchicallyTrackedDict()
                    args0.update(args[0])
                    args0._accessed_keys.clear()  # Note: Very frustrating debugging this as VSCode will automatically query all keys.
                return args0, *args[1:]

            def post_forward_hook(module, args, ret):
                if not isinstance(ret, dict):
                    # raise ValueError(f"BaseModules should return a dict-like object.")
                    src.warn("BaseModules should return a dict-like object.")
                if return_violations := [
                    x for x in (set(ret.keys()) - module.produced_keys())
                ]:
                    # raise ValueError(f"{module_name} returned unexpected keys: {', '.join([str(x) for x in return_violations])}")
                    src.warn(
                        f"{module_name} ({module.__class__}) returned unexpected keys: {', '.join([str(x) for x in return_violations])}"
                    )
                if args and isinstance(args[0], HierarchicallyTrackedDict):
                    # Yes, this should really always be a HierarchicallyTrackedDict, but for development we'll allow the alternative
                    # as we want to print out all issues in as few passes as possible -- there'll have been a warning as to args0 not being a dict.
                    if access_violations := [
                        x
                        for x in (args[0]._accessed_keys - module.consumed_keys())
                        if not x.is_meta_data
                    ]:
                        # raise ValueError(f"{module_name} accessed unexpected keys: {', '.join([str(x) for x in access_violations])}")
                        src.warn(
                            f"{module_name} ({module.__class__}) accessed unexpected keys: {', '.join([str(x) for x in access_violations])}"
                        )
                    args[0].pop_hierarchy()

            return pre_forward_hook, post_forward_hook

        hook_handles = []
        for child_name, basemodule in relevant_basemodules.items():
            pre_hook, post_hook = named_forward_hooks(child_name)
            hook_handles.append(basemodule.register_forward_pre_hook(pre_hook))
            hook_handles.append(basemodule.register_forward_hook(post_hook))
        return hook_handles

    @rank_zero_only
    def dkcc_unwrap_forwards(self, hook_handles):
        for handle in hook_handles:
            handle.remove()

    def train_step(self, batch: OutputsType, step: int) -> OutputsType:
        if not self.dynamic_key_contract_checked:
            hook_handles = self.dkcc_wrap_forwards()
        ret = self.forward(batch, step)
        if not self.dynamic_key_contract_checked:
            # Remove the wrappers and never check again.
            self.dkcc_unwrap_forwards(hook_handles)
            self.dynamic_key_contract_checked = True

        return ret

    def val_step(self, batch: OutputsType, step: int) -> OutputsType:
        return self.forward(batch, step)

    def test_step(self, batch: OutputsType, step: int) -> OutputsType:
        return self.forward(batch, step)

    def log_viz(
        self,
        main_module: "MainModule",
        prefix: str,
        gt: OutputsType,
        pred: OutputsType,
        max_images: int = 16,
    ):
        pass
