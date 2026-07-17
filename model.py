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

        - run Residual(fn), and
          the output is a torch.Tensor that represents
          the changes between x and fn(x).
          whatever you converted x into, this class will
          give you the residual difference
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

def Upsample(dim: int, dim_out: int | None = None) -> nn.Sequential:
    """
    preconditions:
        - dim is the input channel count, > 0
        - dim_out is the output channel count, if you want
        - not specifying dim_out causes dim channels to be kept
    postconditions:
        - returns nn.Sequential that doubles H and W using nearest neighbor
        - converts dim -> dim_out unless you didn't specify it
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim: int, dim_out: int | None = None) -> nn.Sequential:
    """
    preconditions:
        - dim is the input channel count > 0
        - input the tensor's height and width
          at when you called this, both must be even
          so i can halve it easily
        - dim_out could be specified if you want
    postconditions:
        - returns nn.Sequential that halves height and width
        - 2x2 blocks of height and width are moved into channels,
          and those channels are then converted to dim_out or
          dim if you didn't specify dim_out
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -1 * embeddings)
        embeddings = time[: None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
        return embeddings
