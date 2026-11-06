"""路线 A1：把自研实现注册到 ``aten::gelu``。"""
from __future__ import annotations

import json
import os

import torch

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>
CALL_COUNT = {"gelu": 0, "none": 0, "tanh": 0}
_COUNT_BASE = os.environ.get("A1_GELU_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None
_LIB = None
_REGISTERED_KEY = None
_COUNTERS = []
_SUPPORTED_PLATFORMS = {
    "kunlunxin", "p800-kunlunxin", "p800-kunlunxin-gems5",
    "p800-liyanz-gelu-local",
}


def _load_impl():
    try:
        from .kernel.triton_level import gelu_triton
    except ImportError:
        from kernel.triton_level import gelu_triton
    return gelu_triton


def _bump(approximate: str) -> None:
    CALL_COUNT["gelu"] = CALL_COUNT.get("gelu", 0) + 1
    CALL_COUNT[approximate] = CALL_COUNT.get(approximate, 0) + 1
    for counter in _COUNTERS:
        counter["n"] = counter.get("n", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE, encoding="utf-8") as file:
                counts = json.load(file)
        except (OSError, ValueError):
            pass
        counts["gelu"] = counts.get("gelu", 0) + 1
        counts[approximate] = counts.get(approximate, 0) + 1
        with open(COUNT_FILE, "w", encoding="utf-8") as file:
            json.dump(counts, file)


def gelu_aten(
    self: torch.Tensor, approximate: str = "none",
) -> torch.Tensor:
    _bump(approximate)
    return _load_impl()(self, approximate)


def register_a1(
    dispatch_key: str = "CUDA", counter: dict | None = None,
    impl: str = "triton", platform: str | None = None,
) -> torch.library.Library:
    """注册 ``aten::gelu`` 并保持 Library 生命周期。"""
    global _LIB, _REGISTERED_KEY
    if dispatch_key != "CUDA":
        raise RuntimeError(
            f"gelu P800 expects dispatch_key='CUDA', got {dispatch_key!r}"
        )
    if impl != "triton":
        raise RuntimeError(f"gelu currently supports impl='triton', got {impl!r}")
    if platform is not None and platform not in _SUPPORTED_PLATFORMS:
        raise RuntimeError(
            f"unknown platform={platform!r}; supported: "
            f"{sorted(_SUPPORTED_PLATFORMS)}"
        )
    if counter is not None and counter not in _COUNTERS:
        _COUNTERS.append(counter)
    if _LIB is not None:
        if dispatch_key != _REGISTERED_KEY:
            raise RuntimeError(
                f"gelu already registered for {_REGISTERED_KEY}, "
                f"cannot register again for {dispatch_key}"
            )
        return _LIB

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("gelu", gelu_aten, dispatch_key)
    _LIB = lib
    _REGISTERED_KEY = dispatch_key
    return lib


register = register_a1
