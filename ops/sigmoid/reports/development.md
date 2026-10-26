# aten.sigmoid 开发报告

## 1. 环境

首测环境为昆仑芯 P800；主机物理 2 号卡独占映射到容器 `cuda:0`。
容器实测为 Python 3.10.18、PyTorch 2.9.0+cu129、Triton 模块 3.6.0、
FlagGems 5.3.4.post1.dev12。`p800-liyanz-local` 环境锁核对全部通过；
该别名仅描述个人单卡容器，不进入 PR，也不冒充官方 5.0 锁。

## 2. 定义

接口为 `aten::sigmoid(Tensor self) -> Tensor`，逐元素计算
`1/(1+exp(-x))`。fp16、bf16、fp32 在 fp32 中间精度计算，输出 dtype
与浮点输入一致。测试包含 0D/1D/2D/3D/4D、空张量、三类非连续布局、
两组 launch 阈值前后、Transformer 激活、IEEE 特殊值和饱和值。

## 3. 实现

采用 A1，在 profile 指定的 dispatch key 上注册。P800 后端使用 FlagGems
`pointwise_dynamic` 与仓库验证过的昆仑 CodeGenConfig；通用后端保留显式
尾块 mask 和大张量 block 65536。两者均无 autotune，并在
`torch_device_fn.device` 上下文中启动。不支持的 dtype/layout 使用 CompositeExplicitAutograd redispatch，
避免注册后递归。

## 4. 验证

kernel 层 75 个用例通过，覆盖 19 个 shape、三种优化 dtype、三种回退
dtype 和多布局；A1 op 层命中 9 次，`counter["n"]` 契约同步通过；应用层
在 `2×64×4096` 上分别验证 fp16/bf16/fp32，共命中 3 次且输出一致。
CPU 黄金参考回归 93/93，最差 abs/rel 为 1.192e-7/1.680e-7。

## 5. 性能

口径为每个组合独立子进程、20 次 warmup、100 次计时并显式同步。
4 个尺寸 × 3 个 dtype × 自研/FlagGems/原生共 36 组全部完成。结果不能用
单一倍率概括：小 fp32 输入受启动开销影响，自研最差为原生 9.630×；
`2×64×4096 fp32` 自研为原生 0.518×，`2048×2048 bf16` 为 0.559×。
当前 A1 直调 71.651 us、注册路径 86.013 us，分发开销 14.362 us（20.0%）。

## 6. 风险

- 当前只完成 P800 首测，其他五个平台仍需逐平台 profile 验证和必要适配。
- 当前使用个人 FlagGems 5.3 开发容器，不等同于官方 5.0 profile；报告已
  携带精确版本，后续仍需官方共享环境复验。
- 硬件级厂商 API 未验证，不在本次 Triton 级实现中伪造交付。
- `perf_compare` 为 0 FAIL / 1 NEW，但 NEW 尚无历史值，不能证明无回归。

## 7. 后续

P800 功能、精度和范围证据可以进入代码评审；后续由负责人确认首个性能
基线，并对小 fp32 场景考虑原生路由阈值。其他五个平台必须逐平台形成
同类证据，不能由 P800 结果外推。
