"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic Types
from typing import (
    Any,  # noqa: F401
    Callable,  # noqa: F401
    Dict,  # noqa: F401
    Iterable,  # noqa: F401
    Iterator,  # noqa: F401
    List,  # noqa: F401
    Literal,  # noqa: F401
    MutableMapping,  # noqa: F401
    NamedTuple,  # noqa: F401
    NewType,  # noqa: F401
    Optional,  # noqa: F401
    Sequence,  # noqa: F401
    Set,  # noqa: F401
    Sized,  # noqa: F401
    Tuple,  # noqa: F401
    Type,  # noqa: F401
    TypeVar,  # noqa: F401
    Union,  # noqa: F401
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import (
    Bool,  # noqa: F401
    Complex,  # noqa: F401
    Float,  # noqa: F401
    Inexact,  # noqa: F401
    Int,  # noqa: F401
    Integer,  # noqa: F401
    Num,  # noqa: F401
    Shaped,  # noqa: F401
    UInt,  # noqa: F401
)

# Config type
from omegaconf import DictConfig  # noqa: F401

# PyTorch Tensor type
from torch import Tensor  # noqa: F401

# Runtime type checking decorator
from typeguard import typechecked as typechecker  # noqa: F401
