#!/usr/bin/env python3
"""aten.gelu 一键运行 kernel、op、C1 应用三层验证。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OP_DIR))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _load_test(name: str):
    path = OP_DIR / "test" / f"{name}_level.py"
    spec = importlib.util.spec_from_file_location(f"gelu_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from common.device import load_profile
    profile = load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin-gems5")
    print(f"[gelu] {profile.summary()}")
    for label, name in (
        ("1/3 kernel 层", "kernel"),
        ("2/3 op 层", "op"),
        ("3/3 C1 应用层", "framework"),
    ):
        print("\n" + "-" * 60)
        print(label)
        result = _load_test(name).run(profile)
        print(f"  => PASS {result}")
    print("\n[gelu] 三层验证通过。")


def perf_cases(profile):
    from common.perf import PerfCase

    def make(approximate: str):
        def _make(current_profile):
            import torch
            try:
                from ops.gelu.kernel.triton_level import gelu_triton
            except ImportError:
                from kernel.triton_level import gelu_triton
            x = torch.randn(
                2, 64, 4096, dtype=torch.bfloat16,
                device=current_profile.torch_device,
            )
            return lambda: gelu_triton(x, approximate)
        return _make

    def bandwidth(elapsed_ms):
        elements = 2 * 64 * 4096
        return {"GBps": elements * 2 * 2 / elapsed_ms / 1e6}

    del profile
    return [
        PerfCase(
            "ops.gelu.none.triton", group="ops", level="kernel",
            make_fn=make("none"), derived=bandwidth,
        ),
        PerfCase(
            "ops.gelu.tanh.triton", group="ops", level="kernel",
            make_fn=make("tanh"), derived=bandwidth,
        ),
    ]


if __name__ == "__main__":
    main()
