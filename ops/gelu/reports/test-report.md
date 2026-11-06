# 算子测试报告：aten.gelu

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `aten.gelu` / A1 / Triton |
| 目标设备 / 日期 | Kunlunxin P800 / 2026-11-06 |
| 结论 | C1 范围通过；Must 项已完成 |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel | 19 shape、3 dtype、2 mode、布局/特殊值/错误 | `test/kernel_level.py` | PASS，148 例 |
| A1 op | 注册、钉选、调用计数、原生错误语义 | `test/op_level.py` | PASS，14 次拦截 |
| C1 应用 | mini Transformer MLP | `test/framework_level.py` | PASS，6/6 |
| 黄金 | 3 seed、声明式矩阵 | `script/gen_golden.py` + `check_accuracy.py` | PASS，360/360 |
| 性能 | 4 shape × 3 dtype × 2 mode × 3 impl | `script/bench_matrix.py` | 72/72 完成 |
| dispatch | 注册与直调对照 | `script/bench_dispatch.py` | 两模式完成 |

一键入口：标准双卡环境用 `p800-kunlunxin-gems5`；本次物理 2 号卡单卡容器用本机等价 profile `p800-liyanz-gelu-local`。

## 2. 环境

官方 FlagGems 5.0 容器；Python 3.10.18、PyTorch 2.9.0、Triton 3.0.0、FlagGems 5.0.0、XPU-RT 5.0.21。软件锁检查全部通过，独占物理 XPU 2。

## 3. 精度结果

| dtype | 自研 vs fp32 参考 | 原生 | FlagGems |
|---|---|---|---|
| float16 | PASS，容差 `1e-2` | 参考对照通过 | tanh 大幅值存在 NaN 风险 |
| bfloat16 | PASS，容差 `1e-2` | 参考对照通过 | tanh 大幅值存在 NaN 风险 |
| float32 | PASS，容差 `1e-5` | 参考对照通过 | tanh 大幅值存在 NaN 风险 |

自研黄金 360/360 通过，最差 abs `1.221e-04`（Transformer float16/tanh）。最终 kernel 一键复测最差 abs 同为 `1.221e-04`。NaN mask、空张量、确定性和输入敏感性均通过。

## 4. 性能结果

大尺寸 exact 自研为原生延迟的 `0.720–0.737×`；大尺寸 tanh 为 `1.025–1.095×`，但为 FlagGems 延迟的 `0.390–0.442×`。Transformer tanh 自研也快于 FlagGems。小尺寸则因固定启动成本慢于原生，详见 [performance.md](performance.md)。

dispatch 增量：exact `7.557 μs`（14.0%），tanh `8.763 μs`（15.8%），各 120 次调用。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 只在 P800 跑过实机矩阵 | 不构成六平台完成证据 | 后续逐平台执行同一矩阵 |
| 2 | 小输入性能未达原生 | 实际小激活可能回退更优 | 本次记录为限制；第二阶段加入有证据的路由 |
| 3 | FlagGems tanh 大值 NaN | 其快慢不能脱离精度比较 | 已记录，自研已修复 |
| 4 | 未跑 C2/真实模型 | 不可声称生产引擎覆盖 | 保持 C1 标记 |

## 6. 结论

Must 项中的算子结构、reference、黄金、三层测试、性能对照、注册证据和报告均已完成。当前结论严格限定为 P800 forward C1 交付；跨平台、backward 与 C2 不在本结论内。

## 7. 验收清单

| Must 项 | 结果 | 证据 |
|---|---|---|
| 精度必测矩阵 | PASS | 3 dtype、2 mode、19 shape，含边界与特殊值 |
| 确定性与输入敏感性 | PASS | kernel 哨兵断言 |
| 黄金回归 100% | PASS | 360/360 |
| 三层测试 | PASS | kernel 148 例、A1 14 次拦截、C1 6/6 |
| 性能门禁与 dispatch | PASS | 首次记录为 NEW、0 FAIL；两模式 dispatch 已测 |
| 报告与目录登记 | PASS | REPORT、五份分册、catalog、生成索引 |
| 包态导入 | PASS | CI 等价扫描 17/17 |
