"""Cross-module figure overlap audit."""
overlaps = []

# M1 fig3_detectability panel C = "Pairwise Discrimination" (AUROC matrix)
# vs M3 fig1 = pairwise distinguishability AUROC heatmap
overlaps.append({
    "M1图": "fig3_detectability (C. Pairwise Discrimination)",
    "M2/M3图": "M3 fig1 (pairwise AUROC heatmap)",
    "重合度": "高 - 都是两两诊断 AUROC",
    "建议": "保留 M3 fig1（有 Mantel 收敛效度增量），M1 fig3-C 若与 M3 相同则 M1 引用 M3 图",
})

# M1 ed_fig2 = Dendrogram + MDS (clustering + 2D embedding)
# vs M3 fig2 = UMAP geometry
overlaps.append({
    "M1图": "ed_fig2 (Dendrogram + MDS embedding)",
    "M2/M3图": "M3 fig2 (UMAP universe)",
    "重合度": "中 - 都是障碍空间嵌入，M1 用 MDS 障碍级，M3 用 UMAP 用户级",
    "建议": "保留 M3 fig2（用户级、信息更多），M1 ed_fig2 或 M3 fig2 二选一放主图",
})

# M1 ed_fig3 = t-SNE small multiples
# vs M3 fig2 UMAP
overlaps.append({
    "M1图": "ed_fig3 (t-SNE small multiples)",
    "M2/M3图": "M3 fig2 (UMAP)",
    "重合度": "中 - 都是降维可视化",
    "建议": "保留 M3 UMAP（更好），M1 t-SNE 作为补充或删除",
})

# M1 fig3 = detectability (AUROC vs control) + M2 none
# M3 fig1 = pairwise (disorder vs disorder) - 互补不重合
# 但 M1 fig3-C pairwise vs M3 fig1 pairwise 重合 (above)

# M2 fig4 causal DID + M2 fig5 cross-disorder DID
# vs M1 nothing (M1 无 DID) - 无重合

# M2 fig1 trajectory + event study - 无重合 (M1 无纵向)
# M2 fig2 changepoints - 无重合
# M2 fig3 subtypes - 无重合

# M3 fig3 networks - 无重合 (M1 无网络)
# M3 fig4 comorbidity - 无重合 (M1 无共病)

# M1 fig4 specificity (特征特异性) vs M2 ed_fig3 predictors (亚型预测因子)
overlaps.append({
    "M1图": "fig4_specificity (特征分类/特异性)",
    "M2/M3图": "M2 ed_fig3 (subtype predictor importance)",
    "重合度": "低 - M1 是跨诊断特征特异性，M2 是亚型预测因子，不同目标",
    "建议": "保留两者（不同问题）",
})

# M1 fig1 atlas (效应量热图) vs M3 fig3 (网络拓扑) - 不同
# M1 fig2 pfactor vs M2/M3 - 无重合

print("=== 重合核查结果 ===")
for o in overlaps:
    print(f"\n[{o['M1图']}]")
    print(f"  ↔ {o['M2/M3图']}")
    print(f"  重合度: {o['重合度']}")
    print(f"  建议: {o['建议']}")

print("\n=== 无重合的 M2/M3 图（保留重点）===")
print("M2: fig1 轨迹/事件研究、fig2 拐点、fig3 亚型、fig4 因果、fig5 cross-disorder DID")
print("M3: fig3 网络、fig4 共病（均无 M1 对应）")
