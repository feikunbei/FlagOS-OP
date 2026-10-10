# 算子总体报告：aten.sigmoid

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`，见 `reference.py` |
| 路线 / 级别 | A1 / Triton 级 |
| 首测设备 | 昆仑芯 P800，物理 2 号卡映射为容器 `cuda:0` |
| 日期 / 状态 | 2026-10-10 / P800 功能验证完成，性能基线待建立 |

## 实现矩阵

| 级别 | 状态 | 文件 |
|---|---|---|
| torch 级 | 已实现 | `kernel/torch_level.py` |
| Triton 级 | 已实现并通过 P800 真机验证 | `kernel/triton_level.py` |
| 硬件级 | 未实现，不在本次范围 | `kernel/hardware_level/BUILD.md` |

## 验证矩阵

| 层级 | 入口 | 当前结果 |
|---|---|---|
| kernel | `test/kernel_level.py` | PASS，24 cases，哨兵通过 |
| op/A1 | `test/op_level.py` | PASS，命中 5 次，整数回退通过 |
| 应用 | `test/framework_level.py` | PASS，命中 1 次，输出一致 |
| 黄金数据 | `script/gen_golden.py`、`check_accuracy.py` | PASS，57/57 |
| 性能 | `script/bench_perf.py` | 已测三方；历史门禁基线待建立 |

## 关键数字

| 指标 | 结果 |
|---|---:|
| 黄金最差 abs / rel | 1.192e-7 / 1.669e-7 |
| 自研 / FlagGems / 原生延迟 | 0.070173 / 0.060374 / 0.048589 ms |
| A1 分发开销 | 7.874 us（11.2%） |

结论：P800 正确性 Must 项全绿；由于这是新算子且尚无仓库性能基线，性能
回归门禁暂不能判定，建立基线后才可标记最终可交付。
