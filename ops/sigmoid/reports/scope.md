# aten.sigmoid 应用范围说明

> 本文区分“自研优化”“原生回退”和“不在本次交付范围”，避免把单一
> shape/dtype 的通过结果外推为完整支持。P800 最终实测数字以同目录测试
> 与性能报告为准。

## 1. 接口范围

| 接口 | 状态 | 说明 |
|---|---|---|
| `aten::sigmoid(Tensor self) -> Tensor` | 自研优化 | 本次交付目标，推理 forward |
| `aten::sigmoid_` | 不在范围 | 原地语义需独立注册和别名分析 |
| `aten::sigmoid.out` | 不在范围 | out 参数语义需独立实现 |
| `aten::sigmoid_backward` | 不在范围 | 本次项目面向推理 forward |

## 2. dtype 范围

| dtype | 路径 | 验证要求 |
|---|---|---|
| fp16 | P800 自研 pointwise | 全尺寸正确性 + 性能矩阵 |
| bf16 | P800 自研 pointwise | 全尺寸正确性 + 性能矩阵 |
| fp32 | P800 自研 pointwise | 全尺寸正确性 + 性能矩阵 |
| int32 | 原生 redispatch | 输出 fp32，语义一致 |
| int64 | 原生 redispatch | 输出 fp32，语义一致 |
| bool | 原生 redispatch | 输出 fp32，语义一致 |
| fp64 | 非优化 | P800 推理栈不作为验收 dtype，不宣称设备支持 |
| complex / sparse / quantized | 不在范围 | 不属于当前 LLM 推理应用范围 |

## 3. 尺寸和布局范围

正确性矩阵覆盖：

- 0D 标量、空张量、1D/2D/3D/4D；
- 极小输入 `1`、`17`；
- 边界 `255/256/257`、`511/512/513`；
- P800 大 tile 边界 `65535/65536/65537`；
- 长尾 `1×14336`、`8×5000`、`7×4097`；
- Transformer 激活 `2×128×768`、`2×64×4096`、
  `1×4×32×512`；
- contiguous、转置、步长切片、expand；
- `-Inf/+Inf/NaN/-0/+0` 以及 Sigmoid 饱和值。

性能矩阵覆盖四档元素规模：

| case | shape | 用途 |
|---|---|---|
| tail_hidden | `1×14336` | 单 token/长隐藏维度，观察启动 floor |
| small_activation | `64×1024` | 常见小批量激活 |
| transformer | `2×64×4096` | LLM `[batch, sequence, hidden]` |
| throughput | `2048×2048` | 大张量吞吐 |

每个 case 分别运行 fp16、bf16、fp32，并与 FlagGems、原生实现比较。
性能用例按项目要求每个组合使用独立子进程，采用 20 次 warmup 和 100 次
正式计时，计时前后同步设备。

P800 实测显示范围差异显著：`2×64×4096 fp32` 和 `2048×2048 bf16`
的自研/原生延迟比分别为 0.518× 和 0.559×；但 `1×14336 fp32` 与
`64×1024 fp32` 分别为 9.630× 和 8.958×。因此当前不能宣称全尺寸普遍
加速。建议应用层优先把大规模 bf16 和已验证的 Transformer fp32 纳入候选
优化范围，小 fp32 保留原生路径；具体阈值需在各平台独立标定。

## 4. 平台范围

| 平台 | 状态 |
|---|---|
| 昆仑芯 P800 | 本文首测平台，使用专用 CodeGenConfig |
| 昇腾、寒武纪、壁仞及其他目标平台 | 尚未验证，不由 P800 结果外推 |

通用 Triton fallback 保留在 `kernel/triton_level.py`，但只有平台完成真实
硬件测试并形成独立报告后，才能标记该平台支持。

本次 P800 首测使用个人 FlagGems 5.3 开发容器和本地环境锁，不等同于官方
`p800-kunlunxin-gems5` 5.0 profile；精确版本见 `reports/development.md`。
本地 profile/lock 不随代码提交，正式基线需在官方共享环境复验后建立。

## 5. 验收入口

```bash
python3 ops/sigmoid/test/kernel_level.py p800-liyanz-local
python3 ops/sigmoid/test/op_level.py p800-liyanz-local
python3 ops/sigmoid/test/framework_level.py p800-liyanz-local
python3 ops/sigmoid/script/gen_golden.py --device cpu
python3 ops/sigmoid/script/check_accuracy.py --impl triton --device cuda:0
python3 ops/sigmoid/script/bench_matrix.py \
  --profile p800-liyanz-local --out /tmp/sigmoid-p800-matrix.json
```
