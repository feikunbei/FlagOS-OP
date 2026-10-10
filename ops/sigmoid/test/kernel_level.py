"""kernel 层：精度、边界、哨兵及原生性能对比。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def _sync(device: str) -> None:
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mlu"):
        import torch_mlu  # noqa: F401
        torch.mlu.synchronize()


def _bench(fn, device: str, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def run(profile):
    import torch

    from kernel.torch_level import sigmoid_torch
    from kernel.triton_level import sigmoid_triton
    from reference import sigmoid_reference

    device = profile.torch_device
    tolerances = {
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (8e-3, 8e-3),
        torch.float32: (2e-6, 2e-5),
    }
    shapes = [(), (1,), (17,), (64, 1024), (1, 14336), (8, 5000),
              (7, 4097)]
    max_abs = 0.0
    max_rel = 0.0
    cases = 0

    for shape in shapes:
        for dtype, (atol, rtol) in tolerances.items():
            x = torch.randn(shape, dtype=dtype, device=device) * 8
            got = sigmoid_triton(x)
            expected = sigmoid_reference(x)
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
            assert got.shape == x.shape and got.dtype == x.dtype
            abs_err = (got.float() - expected.float()).abs()
            if abs_err.numel():
                max_abs = max(max_abs, abs_err.max().item())
                denom = expected.float().abs().clamp_min(1e-6)
                max_rel = max(max_rel, (abs_err / denom).max().item())
            cases += 1

    # 非连续输入、空张量与饱和值。
    view = torch.randn(129, 67, device=device, dtype=torch.float32).t()
    assert not view.is_contiguous()
    torch.testing.assert_close(sigmoid_triton(view), sigmoid_reference(view),
                               atol=2e-6, rtol=2e-5)
    empty = torch.empty((0, 7), device=device, dtype=torch.float16)
    assert sigmoid_triton(empty).shape == empty.shape
    extreme = torch.tensor([-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0],
                           device=device)
    torch.testing.assert_close(sigmoid_triton(extreme),
                               sigmoid_reference(extreme),
                               atol=2e-6, rtol=2e-5)

    # 确定性、输入敏感性与 torch 级交叉验证。
    x = torch.randn(4097, dtype=torch.float32, device=device)
    first = sigmoid_triton(x)
    assert torch.equal(first, sigmoid_triton(x)), "同输入两次调用不一致"
    assert not torch.equal(first, sigmoid_triton(x + 1)), "输出不随输入变化"
    torch.testing.assert_close(first, sigmoid_torch(x), atol=2e-6, rtol=2e-5)

    perf_x = torch.randn(2048, 2048, dtype=torch.float32, device=device)
    triton_ms = _bench(lambda: sigmoid_triton(perf_x), device)
    native_ms = _bench(lambda: torch.sigmoid(perf_x), device)
    return {
        "ok": True,
        "cases": cases + 3,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "sentinel": "deterministic+sensitive",
        "triton_ms": round(triton_ms, 6),
        "native_ms": round(native_ms, 6),
        "speedup_vs_native": round(native_ms / triton_ms, 4),
    }


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(2048, 2048, dtype=torch.float32,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    def bandwidth(latency_ms):
        return {"GBps": 2048 * 2048 * 4 * 2 / latency_ms / 1e6}

    from kernel.triton_level import sigmoid_triton
    return [PerfCase("ops.sigmoid.triton", group="ops", level="kernel",
                     make_fn=make(sigmoid_triton), derived=bandwidth)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
