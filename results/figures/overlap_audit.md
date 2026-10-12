# M1/M2/M3 图表重合核查报告

> 核查日期：2026-08-31
> 目的：识别 M1/M2/M3 图表的内容重合，决定保留重点，避免论文冗余。

---

## 一、重合核查结果

### 高重合（需二选一）

| # | M1 图 | M2/M3 图 | 重合内容 | 建议 |
|---|---|---|---|---|
| 1 | **fig3_detectability 面板 C**（Pairwise Discrimination，两两 AUROC） | **M3 fig1**（45 对 AUROC 热图 + Mantel） | 都是两两诊断 AUROC 矩阵 | 保留 **M3 fig1**（有 Mantel 收敛效度增量、CI），M1 面板 C 从主图移除，正文引用 M3 fig1 |
| 2 | **ed_fig2**（Dendrogram + MDS 嵌入） | **M3 fig2**（UMAP 宇宙图） | 都是障碍空间嵌入可视化 | 保留 **M3 fig2**（用户级 UMAP，信息更丰富），M1 ed_fig2 降为历史记录或删除 |
| 3 | **ed_fig3**（t-SNE small multiples） | **M3 fig2**（UMAP） | 都是降维可视化 | 与上同理，t-SNE 与 UMAP 二选一，建议保留 UMAP |

### 中低重合（保留，目标不同）

| # | M1 图 | M2/M3 图 | 说明 |
|---|---|---|---|
| 4 | fig1_atlas（效应量热图，跨障碍×特征） | M3 fig3（网络拓扑/桥接特征） | 不同：M1 是效应量大小，M3 是网络结构 |
| 5 | fig2_pfactor（因子方向一致性） | — | 无 M2/M3 对应 |
| 6 | fig4_specificity（特征特异性悖论） | M2 ed_fig3（亚型预测因子） | 低重合：M1 跨障碍特异性 vs M2 亚型预测，问题不同 |
| 7 | fig3_detectability 面板 A/B（单障碍 vs 对照 AUROC、临床效用） | — | M2/M3 无对应（M2/M3 是患者内部比较） |
| 8 | ed_fig4 混淆矩阵、ed_fig5 消融、ed_fig6 敏感性 | — | 无对应 |

### 无重合的 M2/M3 图（全部保留）

**M2**（无 M1 对应，纵向专属）：
- fig1 轨迹 + 事件研究（核心发现）
- fig2 拐点分析（PELT）
- fig3 亚型（轨迹 + 分布）
- fig4 因果（DID + placebo + 剂量反应）
- fig5 cross-disorder DID
- 全部 ed 图

**M3**（无 M1 对应）：
- fig3 网络分析
- fig4 共病结构
- ed_fig2 桥接矩阵热图

---

## 二、视觉问题核查（用户反馈）

### 问题 1：水平柱状图过多
以下图使用水平柱状图（barh）：
- M2 fig2-c（per-disorder pre-t0 率）、M2 fig5（cross-disorder DID 森林）
- M2 ed_fig3（预测因子）、M2 ed_fig7（lead-time 分层）
- M3 fig1-b（per-disorder 均值 AUROC）、M3 fig3-a（网络密度）、M3 fig3-b（桥接特征）
- M3 fig4-b（top 共病对）、M3 fig4-c（中间态分数）

→ **问题**：同类 barh 过多，视觉单调。

### 问题 2：子图高而窄
- M3 fig2-b（谱系分离柱状图）、M3 fig3-b（桥接特征）等单柱面板在 3 列布局中过窄

---

## 三、处置方案

### 方案：微调 M2/M3 图
1. **减少 barh**：
   - M2 fig2-c：改为横向点图（dot plot）或保持但加数据标签
   - M2 fig5：森林图保留（是标准形式），但横向改为纵向分面
   - M3 fig1-b：改为箱线/点图呈现 per-disorder 均值分布
   - M3 fig3-a（网络密度）：改为散点（density vs n_users）揭示样本量效应
   - M3 fig3-b（桥接特征）：保留 barh 但加宽
2. **修复高窄子图**：调整 gridspec 的 width_ratios，让单面板子图更宽
3. **多样图类型**：引入箱线图、小提琴图、散点图、热图、点图等

### 保留重点
- M1 fig3-C 从主图移除（被 M3 fig1 取代）
- M1 ed_fig2/ed_fig3 与 M3 fig2 二选一（保留 M3）
- M2/M3 全部其他图保留
