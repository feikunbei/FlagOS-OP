#!/usr/bin/env python3
"""单组合 GELU 性能测试，供独立运行和 bench_matrix 调用。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common.perf import bench  # noqa: E402

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=int, nargs="+", default=[2048, 2048])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                        default="bfloat16")
    parser.add_argument("--approximate", choices=["none", "tanh"],
                        default="none")
    parser.add_argument("--impl", choices=["native", "torch", "triton", "flaggems"],
                        default="triton")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.impl == "native":
        candidate = F.gelu
    elif args.impl == "torch":
        from ops.gelu.kernel.torch_level import gelu_torch
        candidate = gelu_torch
    elif args.impl == "triton":
        from ops.gelu.kernel.triton_level import gelu_triton
        candidate = gelu_triton
    else:
        import flag_gems
        candidate = flag_gems.gelu

    x = torch.randn(tuple(args.shape), dtype=getattr(torch, args.dtype),
                    device=args.device)
    function = lambda tensor: candidate(
        tensor, approximate=args.approximate
    )
    # 先解析/编译一次，随后 bench 自身再执行正式 warmup。
    function(x)
    latency = bench(
        lambda: function(x), warmup=args.warmup, iters=args.iters,
        device=args.device,
    )
    bytes_moved = 2 * x.numel() * x.element_size()
    result = {
        "impl": args.impl,
        "shape": args.shape,
        "numel": x.numel(),
        "dtype": args.dtype,
        "approximate": args.approximate,
        "latency_ms": round(latency, 6),
        "GBps": round(bytes_moved / latency / 1e6, 3),
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"{args.impl} @ {tuple(args.shape)} {args.dtype} "
            f"mode={args.approximate}: {latency:.6f} ms, "
            f"{result['GBps']:.3f} GB/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
