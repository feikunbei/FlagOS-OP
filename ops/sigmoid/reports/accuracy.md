# aten.sigmoid 精度报告

| 覆盖项 | 内容 |
|---|---|
| dtype | 优化：fp16、bf16、fp32；回退：int32、int64、bool |
| shape | 0D～4D、空、19 个尺寸、两组阈值、长尾、Transformer |
| layout | contiguous、转置、步长切片、expand |
| 特殊值 | ±Inf、NaN、±0、-100、-20、-1、1、20、100 |
| 容差 | fp32 2e-6；fp16 1e-3；bf16 8e-3 |

## 结果

| 测试 | 结果 | 最差 abs | 最差 rel |
|---|---|---:|---:|
| kernel 层（75 cases） | PASS | 0 | 0 |
| 黄金回归（93 cases） | PASS | 1.192e-7 | 1.680e-7 |

fp16、bf16 的输出与 fp32 参考转换结果逐位一致；fp32 的最差误差远低于
2e-6 容差。确定性和输入敏感性哨兵均通过。
