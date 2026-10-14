# 第五章任务包一图件数据说明

## 已生成图件

- 图 5-A：`fig5A_wuhou_metro_population_overlay.png` / `fig5A_wuhou_metro_population_overlay.pdf`
- 图 5-B：`fig5B_critic_topsis_candidate_threshold.png` / `fig5B_critic_topsis_candidate_threshold.pdf`

## 数据来源

- WorldPop 2025 100 m人口栅格原始文件：`C:\Users\zhang\Desktop\论文相关\精准数据\武侯区地图\chn_pop_2025_CN_100m_R2025A_v1.tif`
- 武侯区人口密度裁剪栅格：`C:\Users\zhang\Desktop\小论文工作\武侯区人口密度.tif`
- 武侯区行政边界：`D:\trae建模\metro_logistics_qvns_nsga2_package\metro_logistics_qvns_nsga2_package\data\510107.json`
- 武侯区地铁站点：`D:\trae建模\metro_logistics_qvns_nsga2_package\metro_logistics_qvns_nsga2_package\data\武侯区地铁站点_2024年10月_WGS84.geojson`
- 武侯区地铁线路折线：`D:\trae建模\metro_logistics_qvns_nsga2_package\metro_logistics_qvns_nsga2_package\data\武侯区地铁线路折线_2024年10月_WGS84.geojson`
- CRITIC-TOPSIS全量排序：`C:\Users\zhang\Desktop\论文相关\第三章相关\地铁物流转运点_CRITIC_TOPSIS排序结果.xlsx`，sheet：`ranking`
- CRITIC-TOPSIS前17名校验表：`D:\trae建模\metro_logistics_qvns_nsga2_package\metro_logistics_qvns_nsga2_package\data\第三章评价结果\critic_topsis_top17.csv`

## 处理逻辑

1. 图 5-A 使用已裁剪到研究区的武侯区人口密度栅格，采用对数色阶显示人口集聚差异，并叠加 Space-L 地铁拓扑连线、51个地铁站点及CRITIC-TOPSIS前17名候选站点。该裁剪栅格与 WorldPop 2025 100 m人口栅格作为同一人口密度底图来源体系使用。
2. 图 5-B 使用 `地铁物流转运点_CRITIC_TOPSIS排序结果.xlsx` 中 `ranking` sheet 的51个站点全量综合贴近度得分绘制降序柱状图，并以第17名金融城 `0.298044` 作为截断阈值。
3. 输出图件同时保存为 PNG 与 PDF，PNG 适合直接插入 Word，PDF 适合后续矢量排版。

## 可插入正文文本

为了直观呈现候选站点与现实物流需求之间的空间匹配关系，本文进一步将 2025 年 WorldPop 100 m 常住人口栅格、武侯区地铁线网 Space-L 拓扑结构和 51 个地铁站点位置进行叠加，如图 5-A 所示。可以看出，武侯区中东部核心片区人口密度较高，形成了较强的末端物流需求引力；省体育馆、火车南站、高升桥等候选站点位于人口集聚区与多条地铁线路交汇或邻近区域，具备较好的“网络可达性—需求承载性”协同基础。

进一步地，图 5-B 给出了 51 个地铁站点基于 CRITIC-TOPSIS 方法得到的综合贴近度降序排行。本文以第 17 名金融城的综合贴近度得分 0.298044 作为初选截断阈值，将排名前 17 位的站点纳入第二阶段布局优化模型的候选集合。该处理一方面压缩了后续优化模型的决策规模，另一方面保留了网络结构优势较强和站域需求潜力较高的站点，从而降低了由候选集合设置不透明引起的“黑箱感”。
