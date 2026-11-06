"""kernel 层：GELU 模式、尺寸、dtype、布局、特殊值与哨兵。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OP_DIR))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _bench(function, device: str, warmup: int = 20, iters: int = 100) -> float:
    import torch
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        function()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _record_error(got, expected, current: float) -> float:
    import torch
    finite = torch.isfinite(expected.float())
    if not bool(finite.any()):
        return current
    error = (got.float()[finite] - expected.float()[finite]).abs()
    return max(current, error.max().item())


def run(profile):
    import torch

    try:
        from ops.gelu.kernel.triton_level import gelu_triton
        from ops.gelu.reference import gelu_reference
    except ImportError:
        from kernel.triton_level import gelu_triton
        from reference import gelu_reference

    device = profile.torch_device
    tolerances = {
        torch.bfloat16: 1e-2,
        torch.float16: 1e-2,
        torch.float32: 1e-5,
    }
    modes = ("none", "tanh")
    shapes = [
        (), (1,), (17,),
        (255,), (256,), (257,),
        (511,), (512,), (513,),
        (65535,), (65536,), (65537,),
        (64, 1024), (1, 14336), (8, 5000), (7, 4097),
        (2, 128, 768), (2, 64, 4096), (1, 4, 32, 512),
    ]
    max_abs = 0.0
    cases = 0

    for shape in shapes:
        for dtype, tolerance in tolerances.items():
            source = torch.randn(shape, dtype=torch.float32).to(
                device=device, dtype=dtype
            )
            for approximate in modes:
                got = gelu_triton(source, approximate)
                expected = gelu_reference(source, approximate)
                torch.testing.assert_close(
                    got, expected, atol=tolerance, rtol=0.0, equal_nan=True
                )
                assert got.shape == source.shape and got.dtype == source.dtype
                max_abs = _record_error(got, expected, max_abs)
                cases += 1

    for dtype, tolerance in tolerances.items():
        layouts = [
            torch.randn(129, 67, dtype=dtype, device=device).t(),
            torch.randn(128, 130, dtype=dtype, device=device)[::2, 1::3],
            torch.randn(1, 257, dtype=dtype, device=device).expand(8, 257),
        ]
        for source in layouts:
            assert not source.is_contiguous()
            for approximate in modes:
                got = gelu_triton(source, approximate)
                expected = gelu_reference(source, approximate)
                torch.testing.assert_close(
                    got, expected, atol=tolerance, rtol=0.0, equal_nan=True
                )
                max_abs = _record_error(got, expected, max_abs)
                cases += 1

    values = [
        -float("inf"), -100.0, -20.0, -10.0, -5.0, -3.0, -1.0,
        -0.0, 0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0,
        float("inf"), float("nan"),
    ]
    for dtype, tolerance in tolerances.items():
        empty = torch.empty((0, 7), dtype=dtype, device=device)
        special = torch.tensor(values, dtype=dtype, device=device)
        for approximate in modes:
            empty_out = gelu_triton(empty, approximate)
            assert empty_out.shape == empty.shape and empty_out.dtype == dtype
            cases += 1

            got = gelu_triton(special, approximate)
            expected = gelu_reference(special, approximate)
            torch.testing.assert_close(
                got, expected, atol=tolerance, rtol=0.0, equal_nan=True
            )
            assert torch.equal(torch.isnan(got), torch.isnan(expected))
            max_abs = _record_error(got, expected, max_abs)
            cases += 1

    for dtype in (torch.int32, torch.int64, torch.bool):
        source = torch.ones(8, dtype=dtype, device=device)
        try:
            gelu_triton(source, "none")
        except (RuntimeError, NotImplementedError, ValueError):
            cases += 1
        else:
            raise AssertionError(f"{dtype} 应保持原生不支持语义")

    try:
        gelu_triton(torch.ones(8, device=device), "invalid")
    except RuntimeError as error:
        assert "none or tanh" in str(error)
        cases += 1
    else:
        raise AssertionError("非法 approximate 未报错")

    sentinel_x = torch.randn(64, 1024, dtype=torch.bfloat16, device=device)
    for approximate in modes:
        first = gelu_triton(sentinel_x, approximate)
        assert torch.equal(first, gelu_triton(sentinel_x, approximate))
        assert not torch.equal(
            first, gelu_triton(sentinel_x + 1, approximate)
        )

    perf_x = torch.randn(
        2, 64, 4096, dtype=torch.bfloat16, device=device
    )
    triton_ms = _bench(lambda: gelu_triton(perf_x, "tanh"), device)
    native_ms = _bench(
        lambda: torch.nn.functional.gelu(perf_x, approximate="tanh"), device
    )
    return {
        "ok": True,
        "cases": cases,
        "shape_count": len(shapes),
        "modes": list(modes),
        "dtypes": ["float16", "bfloat16", "float32"],
        "fallback_error_dtypes": ["int32", "int64", "bool"],
        "max_abs": max_abs,
        "sentinel": "deterministic+sensitive",
        "triton_ms": round(triton_ms, 6),
        "native_ms": round(native_ms, 6),
        "latency_ratio": round(triton_ms / native_ms, 4),
    }


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin-gems5")))
