# aten.gelu 黄金数据

> wt-2026-11-06-gelu-p800 · wt <liyanze1560495@hotmail.com>

`inputs_spec.yaml` 是输入事实来源，覆盖 exact-erf 与 tanh 近似两种 ATen
模式、三种核心浮点 dtype、20 个声明 shape（19 个唯一 shape）、三个随机种子以及
zeros/large/boundary 特殊值。

```bash
python3 ops/gelu/script/gen_golden.py --device cpu
python3 ops/gelu/script/check_accuracy.py \
  --impl triton --device p800-liyanz-gelu-local
```

生成物位于 `data/` 和 `index.json`，仅用于本地回归，不提交 Git。参考输出
始终由 CPU 上的 `reference.py` 生成；fp16/bf16/fp32 均以 fp32 中间精度
计算后 cast 回输入 dtype。

特殊值采用 GELU 有意义的有限输入：`±100`、`±20`、`±10`、`±5`、
`±3`、`±1` 和 `0`，避免用 dtype 最大值把三次方中间量无意义地推到溢出。
