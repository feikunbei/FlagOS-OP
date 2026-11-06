"""``aten::gelu`` 的 fp32 语义参考实现。"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>
_APPROXIMATIONS = ("none", "tanh")


def _validate_approximate(approximate: str) -> None:
    if approximate not in _APPROXIMATIONS:
        raise RuntimeError("approximate argument must be either none or tanh.")


def gelu_reference(
    x: torch.Tensor, approximate: str = "none",
) -> torch.Tensor:
    """按 ATen 语义计算 GELU，低精度输入在 fp32 中间精度计算。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"gelu expects a Tensor, got {type(x)!r}")
    _validate_approximate(approximate)
    if not x.is_floating_point() or x.dtype == torch.float64:
        return F.gelu(x, approximate=approximate)

    xf = x.float()
    if approximate == "none":
        out = 0.5 * xf * (1.0 + torch.erf(xf / math.sqrt(2.0)))
    else:
        inner = math.sqrt(2.0 / math.pi) * (
            xf + 0.044715 * xf * xf * xf
        )
        out = 0.5 * xf * (1.0 + torch.tanh(inner))
    return out.to(x.dtype)
