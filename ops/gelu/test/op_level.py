"""op 层：包态 A1 注册、两种 approximate 模式和原生错误回退。"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OP_DIR))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def run(profile):
    import torch
    import torch.nn.functional as F

    try:
        from ops.gelu import register_a1
    except ImportError:
        from register import register_a1

    device = profile.torch_device
    valid = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        source = torch.randn(2, 64, 4096, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        for approximate in ("none", "tanh"):
            valid.append((source, approximate,
                          F.gelu(source, approximate=approximate).clone()))

    noncontiguous = torch.randn(129, 67, device=device).t()
    empty = torch.empty(0, 7, device=device)
    for source in (noncontiguous, empty):
        for approximate in ("none", "tanh"):
            valid.append((source, approximate,
                          F.gelu(source, approximate=approximate).clone()))

    counter = {"n": 0}
    library = register_a1(
        profile.dispatch_key, counter=counter, impl="triton",
        platform=profile.name,
    )
    assert library is not None
    tolerance = {
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
        torch.float32: 1e-5,
    }
    for source, approximate, expected in valid:
        output = F.gelu(source, approximate=approximate)
        torch.testing.assert_close(
            output, expected, atol=tolerance[source.dtype], rtol=0.0,
            equal_nan=True,
        )

    valid_calls = len(valid)
    assert counter["n"] == valid_calls

    for dtype in (torch.int32, torch.int64, torch.bool):
        source = torch.ones(8, dtype=dtype, device=device)
        try:
            F.gelu(source)
        except (RuntimeError, NotImplementedError, ValueError):
            pass
        else:
            raise AssertionError(f"{dtype} 应保持原生不支持语义")

    try:
        F.gelu(torch.ones(8, device=device), approximate="invalid")
    except RuntimeError as error:
        assert "none or tanh" in str(error)
    else:
        raise AssertionError("非法 approximate 未报错")

    expected_calls = valid_calls + 4
    assert counter["n"] == expected_calls
    return {
        "ok": True,
        "registered": f"aten::gelu@{profile.dispatch_key}",
        "intercepted_calls": counter["n"],
        "valid_calls": valid_calls,
        "modes": ["none", "tanh"],
        "dtypes": ["float16", "bfloat16", "float32"],
        "fallback_error_dtypes": ["int32", "int64", "bool"],
        "layout": "contiguous+noncontiguous+empty",
    }
