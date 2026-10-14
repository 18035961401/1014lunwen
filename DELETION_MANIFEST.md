# 删除文件清单与原因

- `canzhao`：旧参考图片，与主程序、消融实验和结果可视化无关。
- `figures`：旧算法命名混用流程图与图片资源，不参与当前求解和最终结果可视化。
- `results`：旧 pilot 结果，避免与用户重新运行后的正式结果混淆；脚本会自动新建 results。
- `data/sf78_wuhou_sf_sites_with_demand_scenarios.xlsx`：原始 xlsx 副本，程序读取的是清洗后的 CSV。
- `data/第三章评价结果/地铁物流转运点_CRITIC_TOPSIS排序结果.xlsx`：原始评价 xlsx，程序读取的是 critic_topsis_top17.csv。
- `data/第三章评价结果/critic_topsis_top17.txt`：CSV 已保留，txt 重复。
- `data/critic_topsis结果.txt`：第三章说明性 txt，与主程序读取无关。
- `UPDATE_MANIFEST_ENHANCED_QVNS_NO_BUDGET.md`：旧无预算版本更新清单，与当前无预算 + NSGA-II/QVNS-NSGA-II 双算法版本不一致。
- `QVNS_NSGA2_ENHANCEMENT_CHANGELOG.md`：旧增强版 changelog，已由当前 README 和 docs 替代。

## 已恢复保留

- `data/顺丰站点数据/`：保留原始顺丰站点 xlsx、数据编写说明 docx 与原始 GeoJSON，用于数据追溯、复核和后续重新清洗。主程序默认仍读取 `data/武侯区_顺丰站点_精准清理_20251218_073112.csv`，以保证求解流程稳定。
