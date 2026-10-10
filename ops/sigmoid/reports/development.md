# aten.sigmoid 开发报告

## 1. 环境

首测环境为昆仑芯 P800；主机物理 2 号卡独占映射到容器 `cuda:0`。
容器预装 PyTorch 2.9.0、Triton 3.6.0 和 FlagGems 5.3.4 开发环境，
不在容器内改装依赖。

## 2. 定义

接口为 `aten::sigmoid(Tensor self) -> Tensor`，逐元素计算
`1/(1+exp(-x))`。fp16、bf16、fp32 在 fp32 中间精度计算，输出 dtype
与浮点输入一致。测试包含标量、空张量、非连续输入、非 block 整倍数
以及饱和值。

## 3. 实现

采用 A1，在 profile 指定的 dispatch key 上注册。P800 后端使用 FlagGems
`pointwise_dynamic` 与仓库验证过的昆仑 CodeGenConfig；通用后端保留显式
尾块 mask 和大张量 block 65536。两者均无 autotune，并在
`torch_device_fn.device` 上下文中启动。不支持的 dtype/layout 使用 CompositeExplicitAutograd redispatch，
避免注册后递归。

## 4. 验证

kernel 层 24 个用例通过，最大误差为 0（该层参考与 kernel 在相同设备上
使用同一 fp32 数学式）；A1 op 层命中 5 次且整数回退正确；隔离应用子进程
命中 1 次，注册前后矩阵前向输出一致。CPU 黄金参考回归 57/57，通过时的
最差 abs/rel 分别为 1.192e-7/1.669e-7。

## 5. 性能

口径为 20 次 warmup、100 次计时，显式设备同步，比较自研与原生
`torch.sigmoid`。2048x2048 fp32 的独立进程结果：自研 0.070173 ms、
FlagGems 0.060374 ms、原生 0.048589 ms。自研比原生慢 44.4%，比 FlagGems
慢 16.2%；主要额外成本是强制设备上下文和多平台 facade。64x1024 fp32
A1 直调 70.266 us、注册路径 78.141 us，分发开销 7.874 us（11.2%）。

## 6. 风险

- 当前只完成 P800 首测，其他五个平台仍需逐平台 profile 验证和必要适配。
- 当前软件版本与仓库 lock 版本不同，结果报告必须携带实际环境版本。
- 硬件级厂商 API 未验证，不在本次 Triton 级实现中伪造交付。
- sigmoid 尚无历史性能基线，回归门禁不能判定为 PASS。

## 7. 后续

P800 功能与精度可以进入代码评审；提交前应建立经负责人确认的性能基线，
并继续评估减少 facade/context 开销的方式。其他五个平台逐平台验证后再补
backend 分叉，不能由 P800 结果外推。
