# aten.gelu 硬件级状态

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

本次 P800 交付没有硬件级实现，运行路径是 [../triton_level.py](../triton_level.py) 与 [../backends/kunlunxin.py](../backends/kunlunxin.py)。本目录保留在模板结构中，只用于明确边界，不能作为可编译或可调用的实现。

| 目标 | 状态 | 原因 |
|---|---|---|
| NVIDIA CUDA C++ | 未实现 | 不是本次目标平台；nvcc 产物不能在 P800 执行（known-issues #12） |
| P800 厂商 C++ 委托 | 未实现 | 当前环境未确认等语义、两模式均可用的稳定公开绑定 |
| P800 XTC/xcc 设备码 | 未实现 | 需要厂商 SDK、编译链和独立验收流程 |

若后续增加硬件级实现，必须同时补齐两种 `approximate` 模式、三种 dtype、非整倍数尾块、特殊值、stream 语义以及与当前 360 例黄金数据的对照，不得只替换文件而沿用本次 Triton 证据。
