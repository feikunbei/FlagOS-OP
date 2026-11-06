"""昆仑芯 P800 的 GELU pointwise 实现。"""

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.utils.pointwise_dynamic import (
    pointwise_dynamic,
)
from flag_gems.utils import tl_extra_shim

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>
PLATFORM = "p800-kunlunxin"
_erf = tl_extra_shim.erf
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _gelu_none_pointwise(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _gelu_tanh_pointwise(x):
    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32
    inner = 0.7978845608028654 * x * (1.0 + 0.044715 * x_sq)
    bounded = tl.minimum(tl.maximum(inner, -10.0), 10.0)
    tanh_value = _tanh(bounded)
    tanh_value = tl.where(inner > 10.0, 1.0, tanh_value)
    tanh_value = tl.where(inner < -10.0, -1.0, tanh_value)
    return 0.5 * x * (1.0 + tanh_value)


def gelu_kunlunxin(x, approximate="none"):
    kernel = (_gelu_tanh_pointwise if approximate == "tanh"
              else _gelu_none_pointwise)
    with torch_device_fn.device(x.device):
        return kernel(x)
