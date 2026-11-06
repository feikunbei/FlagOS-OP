# 算子开发报告：aten.gelu

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::gelu` forward |
| 实现路线 / 级别 | A1 / torch + Triton（P800 backend） |
| 目标设备 | `p800-kunlunxin-gems5` |
| 日期 / 状态 | 2026-11-06 / 待评审 |

## 1. 环境配置

独立 GELU 容器使用官方 FlagGems 5.0 镜像，固定物理 XPU 2，在容器内呈现为 `cuda:0`。仓库标准 `p800-kunlunxin-gems5` profile 预期双卡并选择 `cuda:1`，所以本次用忽略、不提交的单卡等价 profile `p800-liyanz-gelu-local` 执行；`scripts/check_env.py` 软件锁全部通过。

| 组件 | 版本 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.0+cu129 |
| Triton | 3.0.0+f69a5406 |
| FlagGems | 5.0.0 |
| torch_xmlir | 1.0.0.1 |
| xtorch_ops | 0.1.2935+50a5d6a4 |
| XPU-RT / 设备 | 5.0.21 / P800 OAM 96 GiB |

## 2. 算子定义

接口为 `aten::gelu(Tensor self, *, str approximate="none") -> Tensor`。`none` 计算

`0.5 * x * (1 + erf(x / sqrt(2)))`

`tanh` 计算

`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`。

参考实现位于 [../reference.py](../reference.py)，低精度输入使用 fp32 中间精度。验证覆盖 19 个唯一 shape、float16/bfloat16/float32、两种模式、非连续/空张量和特殊值；float16/bfloat16 容差 `1e-2`，float32 容差 `1e-5`。

## 3. 实现说明

A1 路线适合标准 ATen 前向算子：`register.py` 在 CUDA dispatch key 拦截 `aten::gelu`，并调用包态导入的 Triton 实现。入口先校验模式和 dtype/layout；不在自研范围的输入用 `CompositeExplicitAutograd` keyset redispatch，避免注册递归。

P800 后端使用 FlagGems `pointwise_dynamic`。exact 分支调用 `tl_extra_shim.erf`；tanh 分支把内部值限制在 `[-10, 10]` 求值后显式恢复 `±1`，修复 P800 5.0 shim 对大输入产生 NaN 的问题。所有 launch 都置于 `torch_device_fn.device` 上下文，未使用 autotune。

| 层级 | 状态 | 说明 |
|---|---|---|
| torch | 完成 | 语义基准/调试实现 |
| Triton | 完成 | 通用入口 + P800 pointwise backend |
| 硬件级 | 未实现 | XTC/xcc 设备码不在当前交付范围 |

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel | PASS | 148 例；19 shape；最终复测最差 abs `1.221e-04`；确定性和敏感性哨兵通过 |
| A1 op | PASS | 10 个有效组合，含错误回退共 14 次拦截 |
| C1 应用 | PASS | mini Transformer MLP，3 dtype × 2 模式，6/6 一致，每例命中一次 |
| 黄金 | PASS | 360/360，最差 abs `1.221e-04` |

## 5. 性能

正式口径为每组合独立子进程、warmup 20、迭代 100。大尺寸 exact 自研比原生快 `1.36–1.39×`；大尺寸 tanh 与原生相差 `2.5–9.5%`，同时比 FlagGems 快 `2.26–2.56×`。小尺寸受约 `50–60 μs` 固定启动开销影响，仍慢于原生。完整 72 组合见 [performance.md](performance.md)。

注册路径相对直调 kernel 的固定增量为 exact `7.557 μs`、tanh `8.763 μs`。精度代价为：自研所有黄金用例通过；FlagGems tanh 的大幅值异常不用于宣称性能优势。

## 6. 已知问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 小尺寸 launch 固定开销高 | 小/尾块 shape 慢于原生 | 后续按 `numel`/mode 路由 |
| 2 | 仅 P800 实机验证 | 不能外推其他五个平台 | 待逐平台移植和复测 |
| 3 | 仅 forward/C1 | 不覆盖 backward 和真实模型引擎 | 明确限制范围 |
| 4 | 非连续输入会连续化 | 不保留原 stride | 文档化，数值已验证 |

## 7. 结论与后续

当前实现满足 A1 forward 的功能、精度、性能记录、黄金数据和 C1 应用证据，可提交评审。后续优先增加小尺寸原生路由，再按六平台任务逐一实现后端；只有真实模型引擎验证完成后才能升级到 C2。

## 附录：复现命令

```bash
python ops/gelu/example.py p800-liyanz-gelu-local
python ops/gelu/script/check_accuracy.py --impl triton --device p800-liyanz-gelu-local
python ops/gelu/script/bench_matrix.py --profile p800-liyanz-gelu-local --warmup 20 --iters 100
python ops/gelu/script/bench_dispatch.py --profile p800-liyanz-gelu-local --approximate none
python ops/gelu/script/bench_dispatch.py --profile p800-liyanz-gelu-local --approximate tanh
```
