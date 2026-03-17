import datetime
import gc
import logging
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager

import safetensors.torch as st
import torch
from packaging import version

import src
from src.utils.config import config_to_primitive
from src.utils.typing import Any, Callable, Tuple

try:
    from lightning_fabric.utilities.rank_zero import _get_rank
except ImportError:
    from lightning_lite.utilities.rank_zero import _get_rank

logpy = logging.getLogger(__name__)


def envars_get_node_info() -> Tuple[int, int, int]:
    """
    Use with caution.
    We handle two cases:
        A: sbatch launched main script:
            NODE_RANK and COUNT_NODE and _get_rank() will return the correct ranks

        B: Single node interactive jobs:
            envars NODE_RANK and/or COUNT_NODE are not setup and _get_rank() returns None
            for the first process if the function is called before the distributed environment
            is setup by lightning

    returns node_rank, num_nodes, rank
    """
    node_rank = os.environ.get("NODE_RANK")
    num_nodes = os.environ.get("COUNT_NODE")
    rank = _get_rank()
    uninitialized_single_node = any(v is None for v in (node_rank, num_nodes))

    if uninitialized_single_node:
        node_rank = 0 if node_rank is None else int(node_rank)
        num_nodes = 1 if num_nodes is None else int(num_nodes)
        logpy.warning(
            f"""Uninitialized single node in process with PID {os.getpid()}:
                      Assuming a single node interactive job.
                      Distributed environment variables NODE_RANK or COUNT_NODE are not setup.
                      Returning node_rank=0 and num_nodes=1"""
        )
        if rank is None:
            rank = 0
            logpy.warning(
                f"""Rank environment variable not setup in process with PID {os.getpid()}:
                          Rank environment variable not setup.
                          Assuming rank zero process for process PID {os.getpid()}. Returning rank=0"""
            )
    else:
        node_rank = int(node_rank)
        num_nodes = int(num_nodes)

    if (
        not all(isinstance(v, int) for v in (node_rank, num_nodes, rank))
        or node_rank > num_nodes
    ):
        msg_dict = {
            n: v
            for v, n in zip(
                (node_rank, num_nodes, rank), ("node_rank", "num_nodes", "rank")
            )
        }
        raise ValueError(f"Inconsistent distributed setup for {msg_dict}")

    return node_rank, num_nodes, rank


def get_distributed_timestamp(num_nodes: int) -> str:
    """
    outputs a distributed timestamp that is shared among all nodes and processes
    """
    now = os.environ.get("DISTRIBUTED_TIMESTAMP", None)
    if now is None:
        if num_nodes != 1:
            raise NotImplementedError(
                "Logging with multinode training requires to set envar DISTRIBUTED_TIMESTAMP"
            )
        else:
            # single Node job -> set environment var DISTRIBUTED_TIMESTAMP
            # subprocesses inherit environment vars
            # assert rank == 0
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            os.environ["DISTRIBUTED_TIMESTAMP"] = now
            logpy.debug(f"Set environment variable DISTRIBUTED_TIMESTAMP={now}")
    else:
        logpy.debug(
            f"Read timestamp {now} from environment variable DISTRIBUTED_TIMESTAMP"
        )
    return now


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = (
        "RANK",
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "SLURM_PROCID",
        "JSM_NAMESPACE_RANK",
    )
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            # print(f"CHECKING RANK_KEY {key}, THE VALUE IS {rank}")
            return int(rank)
    return 0


def get_global_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            # print(f"CHECKING RANK_KEY {key}, THE VALUE IS {rank}")
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path,
    module_name=None,
    ignore_modules=None,
    mapping=None,
    map_location=None,
    load_ema_weights=False,
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    if path.endswith(".safetensors"):
        ckpt = st.load_file(path, device=map_location)
    elif path.endswith(".ckpt"):
        ckpt = torch.load(path, map_location=map_location)
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")

    if load_ema_weights:
        state_dict = {
            k: v
            for k, v in zip(
                ckpt["state_dict"].keys(), ckpt["callbacks"]["EMA"]["ema_weights"]
            )
        }
    else:
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    if mapping is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            if any([k.startswith(m["to"]) for m in mapping]):
                pass
            else:
                state_dict_to_load[k] = v
        for k, v in state_dict.items():
            for m in mapping:
                if k.startswith(m["from"]):
                    k_dest = k.replace(m["from"], m["to"])
                    lrm.info(f"Mapping {k} => {k_dest}")
                    state_dict_to_load[k_dest] = v.clone()
        state_dict = state_dict_to_load

    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def cleanup():
    with torch.cuda.device(get_device()):
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import tinycudann as tcnn

            tcnn.free_temporary_memory()
        except ImportError:
            pass


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


class TimeRecorder:
    _instance = None

    def __init__(self):
        self.items = {}
        self.accumulations = defaultdict(list)
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
        self.enabled = False

    def __new__(cls):
        # singleton
        if cls._instance is None:
            cls._instance = super(TimeRecorder, cls).__new__(cls)
        return cls._instance

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.items[name] = time.time()

    def end(self, name: str, accumulate: bool = False) -> float:
        if not self.enabled or name not in self.items:
            return
        torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        if accumulate:
            self.accumulations[name].append(delta)
        t = delta * self.time_scale
        src.info(f"{name}: {t:.2f}{self.time_unit}")

    def get_accumulation(self, name: str, average: bool = False) -> float:
        if not self.enabled or name not in self.accumulations:
            return
        acc = self.accumulations.pop(name)
        total = sum(acc)
        if average:
            t = total / len(acc) * self.time_scale
        else:
            t = total * self.time_scale
        src.info(f"{name} for {len(acc)} times: {t:.2f}{self.time_unit}")


### global time recorder
time_recorder = TimeRecorder()


@contextmanager
def time_recorder_enabled():
    enabled = time_recorder.enabled
    time_recorder.enable(enabled=True)
    try:
        yield
    finally:
        time_recorder.enable(enabled=enabled)


class HierarchicallyTrackedDictMeta(type):
    def __new__(meta, classname, bases, classDict):
        cls = type.__new__(meta, classname, bases, classDict)

        # Define init that creates the set of remembered keys
        def __init__(self, *args, **kwargs):
            self._accessed_keys = set()
            self._access_hierarchy = []
            return super(cls, self).__init__(*args, **kwargs)

        cls.__init__ = __init__

        def push_hierarchy(self):
            self._access_hierarchy.append(self._accessed_keys)
            self._accessed_keys = set()

        cls.push_hierarchy = push_hierarchy

        def pop_hierarchy(self):
            if not self._access_hierarchy:
                return False
            else:
                self._accessed_keys = self._access_hierarchy.pop().union(
                    self._accessed_keys
                )
                return True

        cls.pop_hierarchy = pop_hierarchy

        # Decorator that stores a requested key in the cache
        def remember(f):
            def _(self, key, *args, **kwargs):
                self._accessed_keys.add(key)
                return f(self, key, *args, **kwargs)

            return _

        # Apply the decorator to each of the default implementations
        for method_name in ["__getitem__", "__contains__", "get"]:
            me = getattr(cls, method_name)
            setattr(cls, method_name, remember(me))

        return cls


class HierarchicallyTrackedDict(dict, metaclass=HierarchicallyTrackedDictMeta):
    pass
