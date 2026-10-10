"""路线 A1：把自研实现注册到 ``aten::sigmoid``。"""
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"sigmoid": 0}
_COUNT_BASE = os.environ.get("A1_SIGMOID_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None
_LIB = None
_REGISTERED_KEY = None


def _load_impl():
    try:
        from .kernel.triton_level import sigmoid_triton
    except ImportError:
        from kernel.triton_level import sigmoid_triton
    return sigmoid_triton


def _bump() -> None:
    CALL_COUNT["sigmoid"] = CALL_COUNT.get("sigmoid", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE, encoding="utf-8") as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["sigmoid"] = counts.get("sigmoid", 0) + 1
        with open(COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f)


def sigmoid_aten(self: torch.Tensor) -> torch.Tensor:
    _bump()
    return _load_impl()(self)


def register_a1(
    dispatch_key: str = "CUDA", *, platform: str | None = None
) -> torch.library.Library:
    """注册一次并保持 Library 生命周期；``platform`` 供统一入口兼容。"""
    del platform
    global _LIB, _REGISTERED_KEY
    if _LIB is not None:
        if dispatch_key != _REGISTERED_KEY:
            raise RuntimeError(
                f"sigmoid already registered for {_REGISTERED_KEY}, "
                f"cannot register again for {dispatch_key}"
            )
        return _LIB
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("sigmoid", sigmoid_aten, dispatch_key)
    _LIB = lib
    _REGISTERED_KEY = dispatch_key
    return lib


register = register_a1
