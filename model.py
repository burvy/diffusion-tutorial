import math
from functools import partial
from typing import Callable, TypeGuard, TypeVar, cast, override

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn

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
          whatever you converted x into, this class will
          give you the data with the residual applied
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
    """
    Encodes a scalar timestep `t` into a vector with length `dim`
    See `README.md` - Sinusoidal Position Embeddings
    """
    def __init__(self, dim: int) -> None:
        """
        preconditions:
            - dim must be even and 4 or above
        postconditions:
            - self.dim holds dim for `forward()` to call
        """
        super().__init__()
        assert dim >= 4 and dim % 2 == 0, "dim must be greater than 4 and even"
        self.dim: int = dim

    @override
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        preconditions:
            - time is a 1d tensor (just the number), and there is 1 timestep per item in batch
        postconditions:
            - returns a tensor with shape (batch, dim), half are sin, half are cos,
              frequency is spaced geometrically per channel
        """
        device = time.device
        half_dim = self.dim // 2
        freq_scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -1 * freq_scale)
        angles = time[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim = -1)

class WeightStandardizedConv2d(nn.Conv2d): # subclasses nn.Conv2d
    """
    Conv2d has a normalized kernel (0 mean 1 variance)
    weight standardization pairs well with groupnorm to stabilize training
    especially for small batch sizes
    """
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        preconditions:
            input is shaped like (batch, in_channels, H, W)
            in_channels is like this layer's channels
        postconditions:
            returns the normal Conv2d output but computed against
            a normalized copy of self.weight, leaving self.weight
            untouched
        """
        eps = 1e-5 if input.dtype == torch.float32 else 1e-3 # no /0, f16 needs looser eps
        weight = self.weight
        # o (output axis) ... (all other axes) o 1 1 1 (collapse everything except o)
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        # we arent estimating anything since we have all the weights, so unbiased=False
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt() # 1 / sqrt(x)
        return F.conv2d( # parent's forward would use raw self.weight but we want normalized
            input, normalized_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )

class Block(nn.Module):
    """
    WeightStandardizedConv2d -> GroupNorm -> FiLM scale/shift -> SiLU
    ResnetBlock stacks this twice per resolution
    """
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        """
        preconditions:
            dim, dim_out are positive counts of channels
            groups divides dim_out evenly (nn.GroupNorm checks this)
        postconditions:
            self.proj: dim -> dim_out, 3x3, padding=1 (preserves H, W)
            self.norm: GroupNorm over dim_out channels
            self.act: SiLU
        """
        super().__init__()
        self.proj: WeightStandardizedConv2d = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm: nn.GroupNorm = nn.GroupNorm(groups, dim_out)
        self.act: nn.SiLU = nn.SiLU() # x * sigmoid(x), looks like ReLU kind of but smooth

    @override
    def forward(self, x: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """
        preconditions:
            x is shaped (batch, dim, H, W)
            scale_shift is a (scale, shift) pair applicable against x
            after projection (batch, dim_out, 1, 1)
            it is FiLM
        postconditions:
            returns shape (batch, dim_out, H, W)
            x isn't mutated in place
        """
        x = cast(torch.Tensor, self.proj(x))
        x = cast(torch.Tensor, self.norm(x))
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = cast(torch.Tensor, self.act(x))
        return x

class ResnetBlock(nn.Module):
    """
    2 blocks and residual connection
    read the README to learn about Residual
    dim and dim_out can differ so res_conv converts them to be legal
    """
    # the * makes time_emb_dim and groups keywords only, so you need to write
    # ResnetBlock(64, 128, time_emb_dim=32) so theres less bugs
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, groups: int = 8) -> None:
        """
        preconditions:
            dim, dim_out are positive counting channels
            time_emb_dim matches the last dim of the time embedding tensor forward() recieves
        postconditions: SiLU -> Linear(time_emb_dim, dim_out * 2)
        None if there is no conditioning
        self.block1: dim -> dim_out
        self.block2: dim_out -> dim_out
        self.res_conv: Identity if dim is dim_out, 1x1 conv when dim becomes dim_out
        keeps h + self.res_conv(x) legal
        """
        super().__init__()
        self.mlp: nn.Sequential | None = ( # produces FiLM parameters
            # maps time embedding to dim_out * 2, scale and shift for each channel
            # bridge from sinusoidal position embeddings
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        self.block1: Block = Block(dim, dim_out, groups=groups)
        self.block2: Block = Block(dim_out, dim_out, groups=groups)
        self.res_conv: nn.Conv2d | nn.Identity = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    @override
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        preconditions:
            x is shaped (batch, dim, H, W)
            time_emb is None if self.mlp is None
            otherwise, time_emb is shape (batch, time_emb_dim) matching self.mlp
        postconditions:
            returns a shape (batch, dim_out, H, W)
            block1 receives conditioning if present
        """
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)  # pyright: ignore[reportAny]
            # trailing 1s stretches to match, determines what features mean
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            # chunk splits fat channel into 2 tensors of batch, dim, 1, 1
            # that is our scale and shift
            scale_shift = time_emb.chunk(2, dim=1)

        # block1 receives conditioning
        h: torch.Tensor = cast(torch.Tensor, self.block1(x, scale_shift=scale_shift))
        # blcok2 doesnt need conditioning
        h = cast(torch.Tensor, self.block2(h))
        # residual connection
        return h + cast(torch.Tensor, self.res_conv(x))
        # this isn't wrapped in Residual because the channel count changes, x has dim, h has dim out
        # you can't add tensors that have different shapes, so res_conv is 1x1 that projects x from dim
        # to dim out so adding is legal
        # when dims match, it is an `nn.Identity()` a layer that returns its input untouched
