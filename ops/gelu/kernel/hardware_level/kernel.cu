// aten.gelu hardware-level implementation is intentionally absent.
//
// This delivery targets Kunlunxin P800 through Triton/XMLIR. NVIDIA CUDA
// device code cannot execute on P800; see BUILD.md and docs/known-issues.md
// issue #12. This file exports no symbol so it cannot be mistaken for a
// validated GELU implementation.
//
// wt-2026-11-06-gelu-p800
// wt <liyanze1560495@hotmail.com>
