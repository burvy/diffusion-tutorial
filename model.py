import math
from inspect import isfunction
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from typing import TypeVar, TypeGuard, Callable, cast, override

T = TypeVar("T")

def exists(x: T | None) -> TypeGuard[T]:
    """
    preconditions:
        - x can be any type T (unconstrained), or None
    postconditions:
        - returns True if x exists
        - returns False if x is None
        - when True, type checker narrows x from T | None to T
    invariants:
        - pure
    """
    return x is not None


def default(val: T | None, d: T | Callable[[], T]) -> T:
    """
    preconditions:
        - val is any type T (unconstrained) or None
        - d is either T or a callable returning T
        - caller must guarantee d's return type is really T,
          as it is unenforced at runtime past the cast()
        - d must run successfully
    postconditions:
        - returns val when val exists
        - otherwise, returns d() if d is callable, else, d itself
        - d is invoked once only on the None branch
    invariants:
        - pure
    """
    if exists(val):
        return val
    return cast(T, d() if callable(d) else d)

class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        """
        preconditions:
            - fn is an nn.Module callable
            - super().__init__() has not already been called
        postconditions:
            - self.fn holds fn being that it is a submodule now
              the resulting Residual is now nn.Module, making it
              safer
        invariants:
            - self.fn is set once
            - Residual owns no parameters, it wraps fn
        """
        super().__init__()
        self.fn: nn.Module = fn

    @override
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """
        preconditions:
            - x is a Tensor that fits forward
            - *args, **kwargs are any extra states forward expects
            - fn(x, *args, **kwargs) must return a tensor that can
              be added to x
        postconditions:
            - returns fn(x, *args, **kwargs) + x
            - only fn's internal buffers may change
        invariants:
            - pure
        """
        return cast(torch.Tensor, self.fn(x, *args, **kwargs) + x)
