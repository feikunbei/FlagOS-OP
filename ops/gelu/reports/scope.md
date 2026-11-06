# aten.gelu 应用范围

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

> 本页把“已验证”“语义回退”“尚未验证”分开，避免用少量样例推断全部输入。

## 已验证范围

| 轴 | 范围 |
|---|---|
| 平台 | 昆仑芯 P800，XPU-RT 5.0.21，FlagGems 5.0 |
| dtype | float16、bfloat16、float32 |
| approximate | `none`、`tanh` |
| rank | 0D、1D、2D、3D、4D |
| numel | 0、1 到 4,194,304；含 255/256/257、65535/65536/65537 边界 |
| layout | contiguous、转置、切片、expand；非连续输入转连续后计算 |
| 数值域 | 正态、均匀、零、边界值、`±20`、`±100`、`±inf`、NaN |
| 应用 | C1 mini Transformer MLP，shape `2×64×4096` |

## 语义与回退

- float16/bfloat16/float32 strided tensor 使用自研 kernel。
- 非连续 strided tensor 会先连续化，输出是连续 tensor；数值和 shape 保持一致，不承诺保留输入 stride。
- float64、非 strided layout 通过 ATen redispatch 回到原生实现。
- int32、int64、bool 保持原生“不支持 GELU”的报错语义。
- 空张量返回同 shape/dtype 的空结果。
- 本实现只覆盖 forward，不声明 autograd/backward 自研能力。

## 尚未覆盖

- 其他五个平台上的实机测试与平台专用优化。
- float64 自研 kernel、复数、量化和稀疏 layout。
- `gelu_backward`、二阶梯度、训练稳定性。
- C2 vLLM/真实模型引擎与长时压力测试。
- 动态 shape 编译缓存上限、并发流和多卡一致性。

这些项目属于后续扩展，不应从当前 C1 结果推断为已支持。
