#!/usr/bin/env python3
"""读取 CPU 黄金并验证 aten.gelu 候选实现。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        "gelu_" + path.replace("/", "_").removesuffix(".py"), HERE / path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry)


def _resolve_device(name: str) -> str:
    if name == "cpu" or name.startswith(("cuda", "xpu", "npu")):
        return name
    from common.device import load_profile
    return load_profile(name).torch_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--golden", default=str(HERE / "goldendata"))
    args = parser.parse_args()

    if args.impl == "reference":
        from ops.gelu.reference import gelu_reference
        candidate = gelu_reference
    elif args.impl == "torch":
        from ops.gelu.kernel.torch_level import gelu_torch
        candidate = gelu_torch
    elif args.impl == "triton":
        from ops.gelu.kernel.triton_level import gelu_triton
        candidate = gelu_triton
    elif ":" in args.impl:
        path, entry = args.impl.split(":", 1)
        candidate = _load(path, entry)
    else:
        raise ValueError(f"unknown impl: {args.impl}")
    device = _resolve_device(args.device)
    golden_root = Path(args.golden)
    input_spec = yaml.safe_load(
        (HERE / "goldendata/inputs_spec.yaml").read_text(encoding="utf-8")
    )
    tolerances = input_spec["tolerance"]["pointwise_abs"]
    index = json.loads((golden_root / "index.json").read_text(encoding="utf-8"))

    passed = failed = 0
    worst = {"case": "", "dtype": "", "mode": "", "abs": 0.0}
    for record in index["files"]:
        data = torch.load(golden_root / record["file"])
        x = data["input"].to(device)
        output = candidate(x, approximate=data["approximate"]).cpu()
        expected = data["expected"]
        assert output.shape == expected.shape
        assert output.dtype == expected.dtype
        finite = torch.isfinite(expected.float())
        abs_error = 0.0
        if bool(finite.any()):
            abs_error = (output.float()[finite] - expected.float()[finite]
                         ).abs().max().item()
        same_nonfinite = torch.equal(
            torch.isnan(output.float()), torch.isnan(expected.float())
        ) and torch.equal(
            torch.isinf(output.float()), torch.isinf(expected.float())
        )
        ok = same_nonfinite and abs_error <= tolerances[data["dtype"]]
        passed += int(ok)
        failed += int(not ok)
        if abs_error >= worst["abs"]:
            worst.update(case=data["case"], dtype=data["dtype"],
                         mode=data["approximate"], abs=abs_error)
        if not ok:
            print("FAIL", record["file"], f"abs={abs_error:.3e}")

    print(
        f"结论: {passed} pass / {failed} fail · 最差 "
        f"{worst['case']} {worst['dtype']} {worst['mode']} "
        f"abs={worst['abs']:.3e}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
