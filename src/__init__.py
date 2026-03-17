import importlib

###  grammar sugar for logging utilities  ###
import logging

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    # Inference-only runtime does not require Lightning.
    def rank_zero_only(fn):
        return fn

logger = logging.getLogger("pytorch_lightning")


def initialize_instance(cls_string, cfg, *args, **kwargs):
    cls = find(cls_string)
    return cls(
        cfg=cls.Config(**cfg) if cfg is not None else cls.Config(), *args, **kwargs
    )


def find(cls_string):
    module_string = ".".join(cls_string.split(".")[:-1])
    cls_name = cls_string.split(".")[-1]
    module = importlib.import_module(module_string, package=None)
    cls = getattr(module, cls_name)
    return cls


@rank_zero_only
def warn(*args, **kwargs):
    logger.warning(*args, **kwargs)


@rank_zero_only
def info(*args, **kwargs):
    logger.info(*args, **kwargs)


@rank_zero_only
def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)
