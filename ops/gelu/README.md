# aten.gelu（昆仑芯 P800）

本目录实现标准 `aten::gelu(Tensor self, *, str approximate="none") -> Tensor` 前向算子，覆盖精确 `erf` 分支和 `tanh` 近似分支。交付路线为 A1：通过 `torch.library` 注册到 `CUDA` dispatch key，在 P800 XMLIR 栈上调用 Triton/FlagGems `pointwise_dynamic` kernel。

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

## 支持范围

| 维度 | 已验证范围 |
|---|---|
| 设备 | Kunlunxin P800，FlagGems 5.0 环境 |
| dtype | `float16`、`bfloat16`、`float32` |
| approximate | `none`、`tanh` |
| shape | 标量、空张量、1D 边界尺寸、非连续输入、Transformer/吞吐尺寸 |
| 集成 | A1 注册；C1 mini Transformer MLP 消费 |
| 不在本次范围 | backward、float64 自研 kernel、整数/布尔、硬件级 XTC kernel、C2 真实模型引擎 |

不支持的 dtype/layout 通过 ATen redispatch 保持原生行为；整数和布尔输入仍按原生语义报错。完整边界见 [reports/scope.md](reports/scope.md)。

## 实现结构

- `reference.py`：fp32 中间精度语义参考。
- `kernel/torch_level.py`：ATen 组合实现。
- `kernel/triton_level.py`：可移植 Triton 入口、布局处理与原生回退。
- `kernel/backends/kunlunxin.py`：P800 `pointwise_dynamic` 后端。
- `register.py`：A1 注册、平台校验与调用计数。
- `test/`：kernel、op、C1 应用三层测试。
- `goldendata/`：19 个唯一 shape、3 个 dtype、2 种模式、3 个 seed 的声明式输入规格。
- `script/`：黄金生成/核验、单项/矩阵/dispatch 性能测试。

P800 5.0 的 `tanh` shim 在大幅值时可能因内部 `exp` 的 `inf/inf` 产生 NaN。本实现先在 `[-10, 10]` 内求值，再恢复数学饱和值，覆盖 `±20`、`±100` 和 `±inf`。

## 复现

在仓库根目录执行。仓库标准双卡环境使用 `p800-kunlunxin-gems5`；本服务器的独立容器只映射物理 2 号卡为逻辑 `cuda:0`，因此使用本机忽略、不提交的等价 profile `p800-liyanz-gelu-local`：

```bash
python scripts/check_env.py --device p800-liyanz-gelu-local
python ops/gelu/script/gen_golden.py --device cpu
python ops/gelu/script/check_accuracy.py \
  --impl triton --device p800-liyanz-gelu-local
python ops/gelu/example.py p800-liyanz-gelu-local
python ops/gelu/script/bench_matrix.py \
  --profile p800-liyanz-gelu-local --warmup 20 --iters 100
python ops/gelu/script/bench_dispatch.py \
  --profile p800-liyanz-gelu-local --approximate none
python ops/gelu/script/bench_dispatch.py \
  --profile p800-liyanz-gelu-local --approximate tanh
```

性能矩阵的每个“尺寸 × dtype × 模式 × 实现”组合在独立子进程中运行，避免编译缓存或注册状态相互污染。

## 当前结论

- kernel 148 例、黄金 360/360、A1 注册和 C1 六组合均通过。
- `2048×2048` exact 分支比原生快约 `1.36–1.39×`；tanh 分支与原生接近，并比 FlagGems 5.0 快 `2.26–2.56×`。
- 小尺寸受约 `50–60 μs` 启动固定成本影响，尚未达到原生延迟，后续适合按 `numel` 和模式做路由。
- 证据等级为 C1，不声称 C2/vLLM 生产链验证。

详细数字见 [REPORT.md](REPORT.md) 与 [reports/](reports/)。
