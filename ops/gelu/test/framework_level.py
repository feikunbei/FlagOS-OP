"""C1 应用层：Transformer MLP 激活真实消费 A1 GELU 注册。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OP_DIR))

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _mini_transformer(device: str, dtype_name: str, approximate: str):
    import torch
    import torch.nn.functional as F
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(20261106)
    x = torch.randn(2, 64, 4096, generator=generator).to(
        device=device, dtype=dtype
    )
    bias = torch.randn(4096, generator=generator).to(
        device=device, dtype=dtype
    ) * 0.02
    gate = torch.randn(2, 64, 4096, generator=generator).to(
        device=device, dtype=dtype
    )
    activated = F.gelu(x + bias, approximate=approximate)
    return (activated * gate).float().mean(dim=-1)


def _child(out_path: str, register: bool, device: str, dispatch_key: str,
           platform: str, dtype_name: str, approximate: str) -> None:
    if register:
        try:
            from ops.gelu import register_a1
        except ImportError:
            from register import register_a1
        counter = {"n": 0}
        register_a1(dispatch_key, counter=counter, platform=platform)
    else:
        counter = {"n": 0}
    output = _mini_transformer(device, dtype_name, approximate)
    import torch
    torch.save(output.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps({
        "calls": counter["n"],
        "dtype": dtype_name,
        "approximate": approximate,
        "shape": [2, 64, 4096],
    }), encoding="utf-8")


def _run_child(register: bool, profile, dtype_name: str, approximate: str):
    import torch
    with tempfile.TemporaryDirectory() as directory:
        output_path = os.path.join(directory, "result.json")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--child", output_path,
            "--torch-device", profile.torch_device,
            "--dispatch-key", profile.dispatch_key,
            "--platform", profile.name,
            "--dtype", dtype_name,
            "--approximate", approximate,
        ]
        if register:
            command.append("--register")
        subprocess.run(command, check=True, cwd=str(ROOT),
                       env=os.environ.copy())
        metadata = json.loads(Path(output_path).read_text(encoding="utf-8"))
        output = torch.load(output_path + ".pt")
        return metadata, output


def run(profile):
    import torch
    results = {}
    total_calls = 0
    for dtype_name, tolerance in (
        ("float16", 1e-3), ("bfloat16", 1e-3), ("float32", 1e-5)
    ):
        for approximate in ("none", "tanh"):
            _, baseline = _run_child(False, profile, dtype_name, approximate)
            metadata, plugged = _run_child(
                True, profile, dtype_name, approximate
            )
            assert metadata["calls"] == 1
            torch.testing.assert_close(
                plugged, baseline, atol=tolerance, rtol=0.0
            )
            key = f"{dtype_name}/{approximate}"
            results[key] = {"calls": 1, "output_match": True}
            total_calls += 1
    return {
        "ok": True,
        "evidence": "C1-mini-transformer-mlp",
        "consumer": "subprocess-transformer-mlp",
        "application_shape": [2, 64, 4096],
        "results": results,
        "calls": total_calls,
    }


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child(
            sys.argv[sys.argv.index("--child") + 1],
            "--register" in sys.argv,
            sys.argv[sys.argv.index("--torch-device") + 1],
            sys.argv[sys.argv.index("--dispatch-key") + 1],
            sys.argv[sys.argv.index("--platform") + 1],
            sys.argv[sys.argv.index("--dtype") + 1],
            sys.argv[sys.argv.index("--approximate") + 1],
        )
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "p800-kunlunxin-gems5")))
