"""应用层：在隔离子进程的推理式前向中验证注册被真实消费。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _app_forward(device: str):
    import torch
    gen = torch.Generator().manual_seed(20261010)
    x = torch.randn(512, 256, generator=gen, dtype=torch.float32).to(device)
    weight = (torch.randn(256, 128, generator=gen, dtype=torch.float32)
              * 0.02).to(device)
    return torch.sigmoid(x) @ weight


def _child(out_path: str, register: bool, device: str,
           dispatch_key: str) -> None:
    import torch
    if register:
        import register as reg
        reg.register_a1(dispatch_key)
    y = _app_forward(device)
    calls = 0
    if register:
        import register as reg
        calls = reg.CALL_COUNT["sigmoid"]
    torch.save(y.cpu(), out_path + ".pt")
    Path(out_path).write_text(json.dumps({"calls": calls}), encoding="utf-8")


def _run_child(register: bool, profile):
    import torch
    with tempfile.TemporaryDirectory() as directory:
        out = os.path.join(directory, "result.json")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", out,
               "--torch-device", profile.torch_device,
               "--dispatch-key", profile.dispatch_key]
        if register:
            cmd.append("--register")
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=os.environ.copy())
        meta = json.loads(Path(out).read_text(encoding="utf-8"))
        tensor = torch.load(out + ".pt", weights_only=True)
    return meta, tensor


def run(profile):
    base_meta, baseline = _run_child(False, profile)
    plug_meta, plugged = _run_child(True, profile)
    del base_meta
    assert plug_meta["calls"] > 0, "应用前向未消费 aten::sigmoid 注册"
    import torch
    torch.testing.assert_close(plugged, baseline, atol=2e-5, rtol=2e-5)
    return {"ok": True, "engine": "subprocess-inference-forward",
            "calls": plug_meta["calls"], "output_match": True}


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child(sys.argv[sys.argv.index("--child") + 1],
               "--register" in sys.argv,
               sys.argv[sys.argv.index("--torch-device") + 1],
               sys.argv[sys.argv.index("--dispatch-key") + 1])
    else:
        from common.device import load_profile
        print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                               else "p800-kunlunxin")))
