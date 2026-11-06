"""GELU 平台后端选择器。"""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache

# wt-2026-11-06-gelu-p800
# wt <liyanze1560495@hotmail.com>


@lru_cache(maxsize=None)
def get_impl(device_type: str):
    """返回当前运行时的专用实现；无专用实现时返回 ``None``。"""
    if device_type == "cuda" and importlib.util.find_spec("torch_xmlir"):
        module = importlib.import_module(f"{__package__}.kunlunxin")
        return module.gelu_kunlunxin
    return None
