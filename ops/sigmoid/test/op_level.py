"""op 层：A1 注册、aten 拦截与回退语义验证。"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import sigmoid_reference

    device = profile.torch_device
    x = torch.linspace(-12, 12, 65535, dtype=torch.float32,
                       device=device).reshape(257, 255)
    expected = sigmoid_reference(x)
    integer = torch.arange(-32, 32, dtype=torch.int32, device=device)
    integer_expected = torch.sigmoid(integer)

    reg.CALL_COUNT["sigmoid"] = 0
    lib = reg.register_a1(profile.dispatch_key, platform=profile.name)
    assert lib is not None

    before = reg.CALL_COUNT["sigmoid"]
    out = torch.sigmoid(x)
    assert reg.CALL_COUNT["sigmoid"] > before, "aten::sigmoid 未被拦截"
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)

    for dtype, atol in ((torch.float16, 1e-3),
                        (torch.bfloat16, 8e-3),
                        (torch.float32, 2e-6)):
        sample = torch.randn(127, 65, dtype=dtype, device=device) * 8
        torch.testing.assert_close(sample.sigmoid(), sigmoid_reference(sample),
                                   atol=atol, rtol=max(atol, 2e-5))

    integer_out = torch.sigmoid(integer)
    assert integer_out.dtype == integer_expected.dtype
    torch.testing.assert_close(integer_out, integer_expected)
    return {
        "ok": True,
        "registered": f"aten::sigmoid@{profile.dispatch_key}",
        "intercepted_calls": reg.CALL_COUNT["sigmoid"],
        "integer_fallback": True,
    }


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
