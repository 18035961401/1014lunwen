# 中文字体修复说明

本版本仅修复图表中文字体乱码问题，不改变 QVNS-NSGA-II 算法逻辑、参数口径和数据读取方式。

## 修改内容

1. `scripts/run_qvns_ablation_experiment.py`
   - 消融实验 SVG 不再固定使用 `Arial`。
   - 新增中文字体回退列表：`Microsoft YaHei`、`SimHei`、`Noto Sans CJK SC`、`Source Han Sans SC`、`PingFang SC`、`Heiti SC`、`WenQuanYi Micro Hei`、`Arial Unicode MS`、`DejaVu Sans`。
   - 支持通过环境变量 `QVNS_SVG_FONT` 指定 SVG 优先字体。
   - 图标题和坐标轴标签改为中文显示。

2. `code/metro_data_plotting_tools.py`
   - Matplotlib 绘图默认使用中文字体候选列表。
   - 支持通过环境变量 `MPL_CJK_FONT` 指定 Matplotlib 优先字体。
   - 设置 `svg.fonttype = none`、`pdf.fonttype = 42`，提高论文插图的中文兼容性。

3. `code/compare_algorithm_metrics.py`
   - 指标对比图同步使用中文字体候选列表。
   - 同步设置 SVG/PDF 字体输出参数。

4. `README_独立运行说明.md` 与 `docs/消融实验说明.md`
   - 新增中文字体配置说明。

## 使用建议

Windows：

```bat
set MPL_CJK_FONT=Microsoft YaHei
set QVNS_SVG_FONT=Microsoft YaHei
```

macOS / Linux：

```bash
export MPL_CJK_FONT="Noto Sans CJK SC"
export QVNS_SVG_FONT="Noto Sans CJK SC"
```

注意：本包不分发字体文件。如果运行或查看图件的机器没有中文字体，PNG 仍可能出现方框。建议论文插图优先使用 SVG 或 PDF。
