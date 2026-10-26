#!/usr/bin/env python3
"""sigmoid 独立性能测试：原生、torch 级和 Triton 级。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from common.perf import bench  # noqa: E402


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(path.replace("/", "_")[:-3],
                                                  HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=int, nargs="+", default=[2048, 2048])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--impl",
                        choices=["native", "torch", "triton", "flaggems"],
                        default="triton")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true",
                        help="最后一行输出可供 bench_matrix 解析的 JSON")
    args = parser.parse_args()
    implementations = {
        "native": torch.sigmoid,
        "torch": _load("kernel/torch_level.py", "sigmoid_torch"),
        "triton": _load("kernel/triton_level.py", "sigmoid_triton"),
    }
    if args.impl == "flaggems":
        import flag_gems
        implementations["flaggems"] = flag_gems.sigmoid
    x = torch.randn(*args.shape, dtype=getattr(torch, args.dtype),
                    device=args.device)
    latency = bench(lambda: implementations[args.impl](x), warmup=args.warmup,
                    iters=args.iters, device=args.device)
    bytes_moved = 2 * x.numel() * x.element_size()
    result = {
        "impl": args.impl,
        "shape": args.shape,
        "numel": x.numel(),
        "dtype": args.dtype,
        "latency_ms": round(latency, 6),
        "GBps": round(bytes_moved / latency / 1e6, 2),
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"{args.impl} @ {tuple(args.shape)} {args.dtype}: "
              f"{latency:.6f} ms, {result['GBps']:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
