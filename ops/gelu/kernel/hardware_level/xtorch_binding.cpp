// aten.gelu vendor C++ binding is intentionally absent.
//
// No stable, public P800 binding with both exact-erf and tanh-approximate
// semantics was accepted for this delivery. The validated implementation is
// kernel/backends/kunlunxin.py. This file exports no pybind symbol and is not
// part of the build.
//
// wt-2026-11-06-gelu-p800
// wt <liyanze1560495@hotmail.com>
