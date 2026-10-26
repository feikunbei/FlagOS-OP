# 算子总体报告：aten.sigmoid

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`，见 `reference.py` |
| 路线 / 级别 | A1 / Triton 级 |
| 首测设备 | 昆仑芯 P800，物理 2 号卡映射为容器 `cuda:0` |
| 日期 / 状态 | 2026-10-26 / P800 应用范围验证完成，历史基线待建立 |

## 实现矩阵

| 级别 | 状态 | 文件 |
|---|---|---|
| torch 级 | 已实现 | `kernel/torch_level.py` |
| Triton 级 | 已实现并通过 P800 真机验证 | `kernel/triton_level.py` |
| 硬件级 | 未实现，不在本次范围 | `kernel/hardware_level/BUILD.md` |

## 验证矩阵

| 层级 | 入口 | 当前结果 |
|---|---|---|
| kernel | `test/kernel_level.py` | PASS，75 cases，19 个 shape，哨兵通过 |
| op/A1 | `test/op_level.py` | PASS，命中 9 次，counter 契约与回退通过 |
| 应用 | `test/framework_level.py` | PASS，三 dtype 共命中 3 次 |
| 黄金数据 | `script/gen_golden.py`、`check_accuracy.py` | PASS，93/93 |
| 性能矩阵 | `script/bench_matrix.py` | PASS，36/36 独立子进程 |
| 性能门禁 | `perf_run.py` + `perf_compare.py` | 0 FAIL，1 NEW（无历史基线） |

## 关键数字

| 指标 | 结果 |
|---|---:|
| 黄金最差 abs / rel | 1.192e-7 / 1.680e-7 |
| 性能矩阵 | 4 shapes × 3 dtypes × 3 implementations = 36 |
| 自研最佳相对原生 | 0.518×（`2×64×4096` fp32） |
| 自研最差相对原生 | 9.630×（`1×14336` fp32） |
| A1 分发开销 | 14.362 us（20.0%） |

结论：P800 正确性和应用范围矩阵全绿，统一性能门禁执行结果为 0 FAIL；
但该 case 首次进入注册表，当前判定为 NEW，需负责人确认首个历史基线后
才能对后续提交进行回归判定。详见 `reports/scope.md`。
