#!/usr/bin/env python3
"""按声明式规格生成 aten.gelu CPU 黄金输入与参考输出。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _load_reference():
    path = HERE / "reference.py"
    spec = importlib.util.spec_from_file_location("gelu_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gelu_reference


def _resolve_device(name: str) -> str:
    if name == "cpu" or name.startswith(("cuda", "xpu", "npu")):
        return name
    from common.device import load_profile
    return load_profile(name).torch_device


def _inject_special(tensor: torch.Tensor, name: str) -> None:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return
    values = {
        "zeros": [0.0],
        "large": [-100.0, -20.0, 20.0, 100.0],
        "boundary": [-10.0, -5.0, -3.0, -1.0,
                     0.0, 1.0, 3.0, 5.0, 10.0],
    }[name]
    count = min(len(values), flat.numel())
    flat[:count] = torch.tensor(values[:count], dtype=tensor.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(HERE / "goldendata/inputs_spec.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(HERE / "goldendata"))
    args = parser.parse_args()

    input_spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    reference = _load_reference()
    device = _resolve_device(args.device)
    output_root = Path(args.out)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "schema": 1,
        "op": input_spec["op"],
        "reference_device": args.device,
        "files": [],
    }

    count = 0
    for case in input_spec["cases"]:
        for shape in case["shapes"]:
            for dtype_name in case["dtypes"]:
                dtype = getattr(torch, dtype_name)
                for approximate in case["approximations"]:
                    for repeat in range(input_spec["seeds_per_case"]):
                        seed = input_spec["seed_base"] + count
                        generator = torch.Generator().manual_seed(seed)
                        factory = (torch.randn if case["distribution"] == "normal"
                                   else torch.rand)
                        x = factory(tuple(shape), generator=generator,
                                    dtype=torch.float32)
                        if case["distribution"] == "uniform":
                            x = x * 2.0 - 1.0
                        x = (x * case["scale"]).to(dtype)
                        for special in case.get("special", []):
                            _inject_special(x, special)
                        expected = reference(
                            x.to(device), approximate=approximate
                        ).cpu()
                        shape_tag = "scalar" if not shape else "x".join(map(str, shape))
                        filename = (
                            f"{count:04d}_{case['name']}_{shape_tag}_"
                            f"{dtype_name}_{approximate}_s{repeat}.pt"
                        )
                        path = data_dir / filename
                        torch.save({
                            "input": x.cpu(),
                            "expected": expected,
                            "shape": shape,
                            "dtype": dtype_name,
                            "approximate": approximate,
                            "seed": seed,
                            "case": case["name"],
                        }, path)
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                        index["files"].append({
                            "file": f"data/{filename}", "sha256_16": digest,
                        })
                        count += 1

    (output_root / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print(f"[gen-golden] {count} 组 → {data_dir}（index.json 已写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
