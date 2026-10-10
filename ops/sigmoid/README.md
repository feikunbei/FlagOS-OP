# aten.sigmoid

`aten::sigmoid(Tensor self) -> Tensor` 的 A1/Triton 实现。当前首个验证
平台为昆仑芯 P800（XPU 通过 XMLIR 呈现为 CUDA），kernel 保持标准
Triton 写法，供后续五个平台复用与适配。

| 项 | 值 |
|---|---|
| 路线 | A1：`torch.library.Library("aten", "IMPL")` |
| 注册名 | `sigmoid` |
| 首测设备 | `p800-kunlunxin`，单卡本地 profile 使用 `cuda:0` |
| 精度 | fp16/bf16/fp32，内部 fp32 计算 |
| P800 后端 | FlagGems pointwise_dynamic + Kunlun CodeGenConfig，无 autotune |
| 通用后端 | mask 保护，大张量 block 65536，无 autotune |
| 回退 | 非 strided 或非支持 dtype 走原生 redispatch |

## 目录入口

- `reference.py`：判卷参考；
- `kernel/torch_level.py`：ATen 组合基线；
- `kernel/triton_level.py`：自研 kernel；
- `kernel/backends/kunlunxin.py`：P800 调度与 codegen 配置；
- `register.py`：A1 注册与命中计数；
- `test/`：kernel、op、应用三层测试；
- `goldendata/` 与 `script/`：声明式黄金数据和性能脚本。

## 复现

在仓库根目录、已激活项目镜像环境后运行：

```bash
python3 ops/sigmoid/test/kernel_level.py p800-liyanz-local
python3 ops/sigmoid/test/op_level.py p800-liyanz-local
python3 ops/sigmoid/test/framework_level.py p800-liyanz-local
python3 ops/sigmoid/example.py p800-liyanz-local

python3 ops/sigmoid/script/gen_golden.py --device cpu
python3 ops/sigmoid/script/check_accuracy.py --impl triton --device cuda:0
python3 ops/sigmoid/script/bench_perf.py --impl triton --device cuda:0
python3 ops/sigmoid/script/bench_perf.py --impl native --device cuda:0
python3 ops/sigmoid/script/bench_perf.py --impl flaggems --device cuda:0
python3 ops/sigmoid/script/bench_dispatch.py --device p800-liyanz-local
```

硬件级昆仑芯 XTC 实现不在本次 Triton 级交付范围内，见
`kernel/hardware_level/BUILD.md`。
