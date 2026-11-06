"""``ops.gelu`` — 标准 ``aten::gelu`` 推理前向实现包。"""
from __future__ import annotations

import importlib

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


def _pkg_import(name: str):
    return importlib.import_module("ops.gelu." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").gelu_reference
    if name == "PLATFORM":
        return _pkg_import("kernel.triton_level").PLATFORM
    raise AttributeError(f"ops.gelu has no attribute {name!r}")


__all__ = ["register_a1", "reference", "PLATFORM"]
