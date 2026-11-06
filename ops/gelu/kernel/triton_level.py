"""``aten::gelu`` 的可移植 Triton 实现。"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>
PLATFORM = "multi-triton"
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_erf = tl_extra_shim.erf
_tanh = tl_extra_shim.tanh


@triton.jit
def _gelu_none_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    # XPU 5.0 shim 的 tanh 在大输入处可能由 exp 的 inf/inf 产生 NaN。
    # 先把真正求值区间钳到 [-10, 10]，再显式恢复数学饱和值。
    bounded = tl.minimum(tl.maximum(inner, -10.0), 10.0)
    tanh_value = _tanh(bounded)
    tanh_value = tl.where(inner > 10.0, 1.0, tanh_value)
    tanh_value = tl.where(inner < -10.0, -1.0, tanh_value)
    y = 0.5 * x * (1.0 + tanh_value)
    tl.store(y_ptr + offsets, y, mask=mask)


def _native_fallback(x: torch.Tensor, approximate: str) -> torch.Tensor:
    """绕过当前 CUDA 注册，保持 fp64/非 strided/非法 dtype 原生语义。"""
    return torch.ops.aten.gelu.default.redispatch(
        _FALLBACK_KEYSET, x, approximate=approximate
    )


def gelu_triton(
    x: torch.Tensor, approximate: str = "none",
) -> torch.Tensor:
    """计算 exact-erf 或 tanh-approximate GELU。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"gelu expects a Tensor, got {type(x)!r}")
    if approximate not in ("none", "tanh"):
        raise RuntimeError("approximate argument must be either none or tanh.")
    if x.layout != torch.strided or x.dtype not in _SUPPORTED:
        return _native_fallback(x, approximate)

    source = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(source)
    n_elements = source.numel()
    if n_elements == 0:
        return out

    try:
        from .backends import get_impl
    except ImportError:
        from kernel.backends import get_impl
    platform_impl = get_impl(source.device.type)
    if platform_impl is not None:
        return platform_impl(source, approximate)

    block_size = (65536 if n_elements >= 65536
                  else max(256, triton.next_power_of_2(n_elements)))
    kernel = (_gelu_tanh_kernel if approximate == "tanh"
              else _gelu_none_kernel)
    with torch_device_fn.device(source.device):
        kernel[(triton.cdiv(n_elements, block_size),)](
            source, out, n_elements, BLOCK_SIZE=block_size
        )
    return out
