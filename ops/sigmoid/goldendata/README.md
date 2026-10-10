# sigmoid 黄金数据

仓库只提交输入规格；`data/` 与 `index.json` 由脚本生成并已在
`.gitignore` 中排除。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device cuda:0
```

规格覆盖 fp16、bf16、fp32，小尺寸、非 256 整倍数尾块、二维长尾，
以及 `-100/-20/0/20/100` 一类饱和值。参考输出来自
`reference.py`，低精度输入统一在 fp32 中计算后转换回输入 dtype。
