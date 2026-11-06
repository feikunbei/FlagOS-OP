# 算子总体报告：aten.gelu

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::gelu` 前向；[reference.py](reference.py) |
| 模式 | `approximate="none"` / `"tanh"` |
| 目标设备 / 路线 | `p800-kunlunxin-gems5` / A1 |
| 日期 / 状态 | 2026-11-06 / 待评审 |
| 证据等级 | C1 mini Transformer MLP；不含 C2 |

## 实现矩阵

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | 完成 | fp32 中间精度基准实现 |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | 完成 | P800 选择 `pointwise_dynamic` 后端 |
| 硬件级 | [kernel/hardware_level/](kernel/hardware_level/BUILD.md) | 未实现 | P800 真设备码需要 XTC/xcc，明确排除在本次范围外 |

## 验证矩阵

| 层级 | 结果 | 关键证据 |
|---|---|---|
| kernel | PASS | 148 例、19 个 shape、3 dtype、2 模式；最终复测最差 abs `1.221e-04` |
| A1 op | PASS | 10 个有效调用；含非法参数和不支持 dtype 后共拦截 14 次 |
| C1 应用 | PASS | Transformer MLP `2×64×4096`，3 dtype × 2 模式，6/6 输出一致 |
| 黄金回归 | PASS | 360/360；最差 abs `1.221e-04` |
| 性能矩阵 | 完成 | 4 shape × 3 dtype × 2 模式 × 3 实现 = 72 组合 |
| dispatch | 完成 | exact `7.557 μs`；tanh `8.763 μs` 注册增量 |

## 关键结果

- 大尺寸 `2048×2048` exact：自研/原生延迟比为 `0.720–0.737×`，即快约 `1.36–1.39×`。
- 大尺寸 tanh：自研为原生的 `1.025–1.095×`，但比 FlagGems 5.0 快 `2.26–2.56×`。
- Transformer tanh：自研 `0.052–0.058 ms`，快于 FlagGems `0.071–0.088 ms`；原生仍更快。
- FlagGems 5.0 tanh 在大幅值输入出现 NaN；自研实现通过稳定饱和处理保持与参考一致。
- 小尺寸主要受启动固定成本限制，当前不宣称全面优于原生。

## 结论与遗留

功能、精度、A1 集成和 C1 使用范围满足当前交付要求，可进入评审。主要遗留是小尺寸路由、C2 真实引擎验证、backward 和其他平台移植；这些均未计入已完成范围。

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 应用范围 | [reports/scope.md](reports/scope.md) |
