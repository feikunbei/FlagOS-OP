#!/usr/bin/env python3
"""测量 aten.gelu A1 注册路径相对直调 kernel 的固定开销。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="p800-kunlunxin-gems5")
    parser.add_argument("--approximate", choices=["none", "tanh"],
                        default="tanh")
    args = parser.parse_args()

    import torch
    from common.device import load_profile
    from common.perf import bench
    from ops.gelu import register_a1
    from ops.gelu.kernel.triton_level import gelu_triton

    profile = load_profile(args.profile)
    x = torch.randn(64, 1024, dtype=torch.float32,
                    device=profile.torch_device)
    direct_ms = bench(
        lambda: gelu_triton(x, args.approximate),
        warmup=20, iters=100, device=profile.torch_device,
    )
    counter = {"n": 0}
    register_a1(profile.dispatch_key, counter=counter, platform=profile.name)
    registered_ms = bench(
        lambda: torch.nn.functional.gelu(x, approximate=args.approximate),
        warmup=20, iters=100, device=profile.torch_device,
    )
    overhead_us = (registered_ms - direct_ms) * 1000.0
    ratio = overhead_us / (direct_ms * 1000.0) * 100.0
    print(f"direct_us={direct_ms * 1000.0:.3f}")
    print(f"registered_us={registered_ms * 1000.0:.3f}")
    print(f"overhead_us={overhead_us:.3f}")
    print(f"overhead_pct={ratio:.1f}")
    print(f"intercepted_calls={counter['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
