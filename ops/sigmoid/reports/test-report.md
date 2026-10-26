# aten.sigmoid 测试报告

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / Triton |
| 目标 | 昆仑芯 P800，单卡物理 2 |
| 日期 | 2026-10-26 |
| 结论 | 范围、功能与精度通过；性能 0 FAIL / 1 NEW |

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel | 19 shape、三优化 dtype、布局、IEEE、回退、哨兵 | PASS，75 cases |
| op | 包态 A1、counter、三优化 dtype、三回退 dtype | PASS，9 次命中 |
| 应用 | `2×64×4096` Transformer 激活，三 dtype | PASS，3 次命中 |
| 黄金 | 阈值、长尾、Transformer、特殊值 | PASS，93/93 |
| 性能矩阵 | 4 shape × 3 dtype × 3 实现，20+100 | PASS，36/36 |
| 统一门禁 | `ops.sigmoid.triton` | OK：0 FAIL / 1 NEW |

环境注意：容器需同时设置 `CUDA_VISIBLE_DEVICES=2`；只设置
`XPU_VISIBLE_DEVICES=2` 时，XMLIR 会报告 8 卡而底层只暴露 1 卡，触发
`device=1, num_gpus=1` 初始化错误。
