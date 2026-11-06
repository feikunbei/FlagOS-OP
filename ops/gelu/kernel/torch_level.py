"""由 PyTorch/ATen 基础算子组成的 GELU 基线实现。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from ..reference import _validate_approximate
except ImportError:
    from reference import _validate_approximate

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def gelu_torch(
    x: torch.Tensor, approximate: str = "none",
) -> torch.Tensor:
    """与 ``aten::gelu`` 同签名的非融合 ATen 组合基线。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"gelu expects a Tensor, got {type(x)!r}")
    _validate_approximate(approximate)
    if not x.is_floating_point() or x.dtype == torch.float64:
        return F.gelu(x, approximate=approximate)

    xf = x.float()
    if approximate == "none":
        result = 0.5 * xf * (1.0 + torch.erf(xf / math.sqrt(2.0)))
    else:
        x3 = xf * xf * xf
        result = 0.5 * xf * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                             (xf + 0.044715 * x3))
        )
    return result.to(x.dtype)
