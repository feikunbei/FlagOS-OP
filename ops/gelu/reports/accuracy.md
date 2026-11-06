# aten.gelu 精度报告

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

> 数据日期：2026-11-06；P800 / FlagGems 5.0 环境。

## 1. 用例覆盖

| case | shape | dtype | 模式 / 分布 | 组数 |
|---|---|---|---|---:|
| small_and_rank | 标量、1、17、64×1024 | fp16/bf16/fp32 | none+tanh / normal+zeros | 72 |
| launch_edges | 255/256/257、511/512/513、65535/65536/65537 | fp16/bf16/fp32 | none+tanh / normal+boundary | 162 |
| long_tail | 1×14336、8×5000、7×4097 | fp16/bf16/fp32 | none+tanh / normal+boundary | 54 |
| transformer | 2×128×768、2×64×4096、1×4×32×512 | fp16/bf16/fp32 | none+tanh / normal | 54 |
| saturation | 257 | fp16/bf16/fp32 | none+tanh / uniform+large+boundary | 18 |
| 合计 | 19 个唯一 shape，3 seed | 3 dtype | 2 模式 | 360 |

kernel 额外覆盖非连续转置/切片/expand、空张量、`±inf`、NaN、整数/布尔报错以及非法 `approximate`。

## 2. 结果

| 实现 | 最差 abs | 相对误差口径 | 判定 | 备注 |
|---|---:|---|---|---|
| 自研 Triton | `1.221e-04` | 近零处不以 rel 作为主判据 | PASS，360/360 | worst：Transformer float16/tanh |
| 原生 ATen | 作为 A1 前置基线 | 与同 dtype 结果比较 | PASS | 3 dtype × 2 mode |
| FlagGems 5.0 | 常规输入可运行 | 大幅值 tanh 无有效误差值 | 条件失败 | `±20/±100/inf` 可产生 NaN |

容差：float16/bfloat16 为 `atol=1e-2, rtol=0`，float32 为 `atol=1e-5, rtol=0`。最终 kernel 一键复测的最差 abs 为 `1.221e-04`。

## 3. 异常定位

| case | 现象 | 根因 | 处理 |
|---|---|---|---|
| tanh saturation | FlagGems 与初版自研在大正/负输入出现 NaN | P800 5.0 tanh shim 的指数表达式可能形成 `inf/inf` | inner 钳到 `[-10,10]` 求值，并按原 inner 恢复 `±1` |
| 非连续输入 | pointwise 后端要求连续内存 | 直接按线性地址会破坏 stride 语义 | 入口显式 `contiguous()`，并验证输出数值 |
| 非浮点 dtype | ATen 原生不支持 GELU | 自研不应静默转换 | redispatch 保持原生错误 |

修复后特殊值的 NaN mask 与 fp32 参考一致，`±inf` 按数学饱和语义返回。
