# 顶级期刊科研绘图风格指南
## 面向计算精神病学 / 社交文本数字表型研究

> 适用方向：基于社交媒体语言行为的 p 因子建模、跨诊断特异性悖论、数字表型精神障碍区分

---

## 一、总体设计哲学

顶级期刊（Nature、Science、Nature Medicine、Nature Human Behaviour、JAMA、Lancet）的图形遵循以下核心原则：

| 原则 | 描述 |
|------|------|
| **信息密度最大化** | 每一像素都必须传递信息，拒绝装饰性元素 |
| **视觉层级清晰** | 数据 > 标注 > 辅助线 > 背景，层次分明 |
| **色彩克制** | 颜色只用于编码信息，不用于美化 |
| **可重现性** | 所有视觉决策必须能被精确描述和复现 |
| **无障碍兼容** | 色盲友好配色（约 8% 男性色盲），灰度打印可读 |

---

## 二、字体规范

### 2.1 字体选择

| 用途 | 推荐字体 | 备注 |
|------|----------|------|
| **主字体（图内文字）** | `Arial` / `Helvetica Neue` | Nature 官方指定无衬线体 |
| **备选主字体** | `Myriad Pro` / `Source Sans Pro` | Science 常用 |
| **数学公式/统计符号** | `Times New Roman` / `STIX Two` | 与正文区分 |
| **代码标注** | `Courier New` / `Consolas` | 基因名、变量名等 |
| **中文补充** | `Source Han Sans`（思源黑体） | 中文图注 |

> ⚠️ **禁止使用**：Comic Sans、Impact、宋体（正文）

### 2.2 字号规范（单位：pt，按 300 DPI 输出）

```
图标题（Figure title）：    8–9 pt，Bold
轴标签（Axis label）：      7–8 pt，Regular
刻度标签（Tick label）：    6–7 pt，Regular
图内标注（Annotation）：    6–7 pt，Regular/Italic
图例文字（Legend text）：   7 pt，Regular
统计标注（p-value 等）：    6–7 pt，Regular
面板标签（A B C D）：       9–10 pt，Bold，左上角
```

### 2.3 字号 Python 参数

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Nature/Science 风格字体全局配置
plt.rcParams.update({
    # 字体族
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    
    # 字号体系
    'font.size':            7,           # 全局基准字号
    'axes.titlesize':       8,           # 子图标题
    'axes.labelsize':       7,           # 轴标签
    'xtick.labelsize':      6,           # x 轴刻度
    'ytick.labelsize':      6,           # y 轴刻度
    'legend.fontsize':      7,           # 图例
    'figure.titlesize':     9,           # 整图标题
    
    # 字重
    'axes.titleweight':     'bold',
    'font.weight':          'regular',
})
```

---

## 三、图形尺寸规范

### 3.1 Nature 系列标准尺寸

| 类型 | 宽度（mm） | 高度（mm） | 说明 |
|------|-----------|-----------|------|
| 单栏图 | 89 | 可变 | 最常用 |
| 1.5 栏图 | 120 | 可变 | 含复杂图例 |
| 双栏图 | 183 | 可变 | 全宽图 |
| 最大高度 | — | 247 | A4 页面限制 |

### 3.2 Science 系列标准尺寸

| 类型 | 宽度（in） | DPI |
|------|-----------|-----|
| 单栏 | 2.25 in (57 mm) | 300 |
| 双栏 | 4.75 in (121 mm) | 300 |
| 全宽 | 7 in (178 mm) | 300 |

### 3.3 Python 尺寸设置

```python
# 转换函数
def mm2inch(mm):
    return mm / 25.4

# Nature 单栏
fig, ax = plt.subplots(figsize=(mm2inch(89), mm2inch(65)))

# Nature 双栏
fig, axes = plt.subplots(1, 3, figsize=(mm2inch(183), mm2inch(60)))

# Science 双栏
fig = plt.figure(figsize=(4.75, 3.5))  # inch

# 输出设置
fig.savefig('figure.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('figure.tiff', dpi=300, bbox_inches='tight',
            facecolor='white', compression='tiff_lzw')
```

---

## 四、配色方案

### 4.1 核心设计原则

- **主色调**：最多使用 4–6 种颜色
- **色盲安全**：优先选用 Okabe-Ito 或 Wong 色板
- **饱和度控制**：避免高饱和原色，偏哑光/粉彩
- **背景**：纯白 `#FFFFFF`，辅助线用极浅灰

### 4.2 推荐配色方案

#### 方案 A：Wong 色盲安全色板（最推荐）

```python
WONG_PALETTE = {
    'black':       '#000000',
    'orange':      '#E69F00',
    'sky_blue':    '#56B4E9',
    'green':       '#009E73',
    'yellow':      '#F0E442',
    'blue':        '#0072B2',
    'vermillion':  '#D55E00',
    'pink':        '#CC79A7',
}

# 列表形式（按使用频率排序）
WONG_8 = ['#0072B2', '#D55E00', '#009E73', '#E69F00',
          '#56B4E9', '#CC79A7', '#F0E442', '#000000']
```

#### 方案 B：Nature Medicine / Nature Human Behaviour 风格

```python
# 精神病学研究常用 — 蓝紫色系为主
NEURO_PALETTE = {
    'primary_blue':    '#2166AC',   # 主色：健康对照
    'secondary_red':   '#D6604D',   # 对比色：患者群体
    'accent_purple':   '#762A83',   # 强调：p 因子
    'neutral_green':   '#1A9850',   # 第三组
    'light_blue':      '#92C5DE',   # 浅色：误差带
    'light_red':       '#F4A582',   # 浅色：误差带
    'gray_major':      '#636363',   # 辅助文字
    'gray_minor':      '#BDBDBD',   # 网格线/辅助线
    'background':      '#FFFFFF',
    'panel_bg':        '#F7F7F7',   # 面板浅背景（慎用）
}
```

#### 方案 C：连续变量热图配色

```python
import matplotlib.colors as mcolors
from matplotlib import cm

# 相关矩阵 / 效应量热图（推荐 RdBu_r）
DIVERGING_CMAP = 'RdBu_r'    # 双向：正负相关
SEQUENTIAL_CMAP = 'viridis'  # 单向：强度/概率
EFFECT_CMAP = 'coolwarm'     # 效应量

# 精神病学 p 因子载荷图推荐
P_FACTOR_CMAP = plt.cm.get_cmap('PuOr', 256)  # 紫-橙双向

# 自定义分段颜色映射（用于诊断分组）
from matplotlib.colors import ListedColormap
DIAGNOSIS_COLORS = ['#2166AC', '#D6604D', '#762A83', '#1A9850', '#E69F00']
diagnosis_cmap = ListedColormap(DIAGNOSIS_COLORS)
```

#### 方案 D：Nature 2024 流行风格（渐变+点图）

```python
# 近期 Nature 高引文章常见的柔和渐变配色
NATURE_2024 = {
    'mdd_blue':     '#4393C3',
    'schiz_red':    '#D6604D',
    'bipolar_purp': '#9970AB',
    'adhd_orange':  '#F4A736',
    'anxiety_green':'#5AAE61',
    'ptsd_gray':    '#878787',
    'p_factor':     '#762A83',   # p 因子专属紫
    'light_fill':   '#F0F0F0',   # 小提琴图填充
}
```

### 4.3 透明度规范

```python
# 散点图重叠区域
scatter_alpha = 0.6

# 误差带（置信区间）
ci_band_alpha = 0.25

# 小提琴图填充
violin_alpha = 0.7

# 轮廓线
edge_alpha = 0.9
```

---

## 五、坐标轴与网格规范

### 5.1 轴线风格

```python
plt.rcParams.update({
    # 轴线
    'axes.linewidth':       0.8,     # 轴框线宽（细）
    'axes.spines.top':      False,   # 隐藏上轴（推荐）
    'axes.spines.right':    False,   # 隐藏右轴（推荐）
    
    # 刻度线
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'xtick.major.size':     3.0,
    'ytick.major.size':     3.0,
    'xtick.minor.size':     1.5,
    'ytick.minor.size':     1.5,
    'xtick.direction':      'out',   # 刻度朝外
    'ytick.direction':      'out',
    
    # 网格
    'axes.grid':            True,
    'grid.color':           '#E0E0E0',
    'grid.linewidth':       0.5,
    'grid.linestyle':       '--',
    'grid.alpha':           0.7,
    'axes.axisbelow':       True,    # 网格在数据下方
})
```

### 5.2 去除顶/右轴的函数

```python
def despine(ax, top=True, right=True, left=False, bottom=False):
    """移除指定轴框，Nature 风格标准操作"""
    if top:    ax.spines['top'].set_visible(False)
    if right:  ax.spines['right'].set_visible(False)
    if left:   ax.spines['left'].set_visible(False)
    if bottom: ax.spines['bottom'].set_visible(False)
    # 同步刻度
    ax.tick_params(top=not top, right=not right,
                   left=not left, bottom=not bottom)
    return ax
```

---

## 六、线条与标记规范

```python
plt.rcParams.update({
    # 线条
    'lines.linewidth':      1.5,     # 主曲线（ROC、学习曲线）
    'lines.markersize':     4.0,     # 散点大小
    'lines.markeredgewidth':0.5,
    
    # 误差棒
    'errorbar.capsize':     3,
    
    # 填充
    'patch.linewidth':      0.8,
    'patch.edgecolor':      'auto',
})

# 常用线型
LINESTYLES = {
    'solid':     '-',
    'dashed':    '--',     # 对照/阈值线
    'dotted':    ':',      # p 值参考线
    'dash_dot':  '-.',     # 次要趋势
}

# 标记形状（用于区分诊断组）
MARKERS = {
    'mdd':      'o',       # 圆形：抑郁
    'schiz':    's',       # 方形：精神分裂
    'bipolar':  '^',       # 三角：双相
    'adhd':     'D',       # 菱形：ADHD
    'anxiety':  'v',       # 倒三角：焦虑
    'control':  'P',       # 加号：健康对照
}
```

---

## 七、统计标注规范

### 7.1 显著性标记

```python
def add_significance(ax, x1, x2, y, p_value, h=0.02, lw=0.8):
    """
    在两组之间添加显著性括号
    
    Parameters
    ----------
    ax : matplotlib Axes
    x1, x2 : float  — 两组的 x 坐标
    y : float        — 括号顶端 y 坐标
    p_value : float  — p 值
    h : float        — 括号高度（数据单位）
    """
    # 显著性符号映射
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    # 绘制括号
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            lw=lw, c='black')
    ax.text((x1+x2)/2, y+h, sig,
            ha='center', va='bottom',
            fontsize=6.5, fontfamily='Arial')

# p 值直接标注格式
def format_pvalue(p):
    if p < 0.0001:
        return 'p < 0.0001'
    elif p < 0.001:
        return f'p = {p:.4f}'
    else:
        return f'p = {p:.3f}'

# 效应量标注（Cohen's d / AUC）
def annotate_effect(ax, x, y, d, auc=None):
    text = f"d = {d:.2f}"
    if auc is not None:
        text += f"\nAUC = {auc:.3f}"
    ax.text(x, y, text, fontsize=6, ha='center',
            style='italic', color='#636363')
```

### 7.2 回归线与置信区间

```python
import numpy as np
from scipy import stats

def plot_regression(ax, x, y, color='#2166AC', label=None, ci=95):
    """绘制回归线 + 置信区间带（Nature 风格）"""
    # 拟合
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    
    # 置信区间
    n = len(x)
    t_crit = stats.t.ppf((1 + ci/100) / 2, df=n-2)
    x_mean = x.mean()
    se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    ci_band = t_crit * se_line
    
    # 绘制
    ax.plot(x_line, y_line, color=color, lw=1.5, label=label, zorder=3)
    ax.fill_between(x_line, y_line - ci_band, y_line + ci_band,
                    color=color, alpha=0.2, zorder=2)
    
    # 标注统计量
    stat_text = f'r = {r:.3f}, {format_pvalue(p)}'
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes,
            fontsize=6.5, va='top', style='italic')
    return slope, r, p
```

---

## 八、研究方向专属图表规范

### 8.1 p 因子载荷图（Factor Loading Heatmap）

```python
import seaborn as sns
import pandas as pd

def plot_p_factor_loadings(loadings_df, ax=None):
    """
    绘制 p 因子载荷热图
    loadings_df: shape (n_symptoms, n_factors)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(mm2inch(89), mm2inch(100)))
    
    # 掩码：极小值不显示
    mask = np.abs(loadings_df.values) < 0.1
    
    sns.heatmap(
        loadings_df,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        mask=mask,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 5.5, 'weight': 'bold'},
        linewidths=0.3,
        linecolor='white',
        cbar_kws={
            'label': 'Factor Loading',
            'shrink': 0.8,
            'pad': 0.02,
        }
    )
    
    ax.set_xlabel('Factors', fontsize=7)
    ax.set_ylabel('Symptom / Language Feature', fontsize=7)
    ax.tick_params(labelsize=6)
    
    # 高亮 p 因子列
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, len(loadings_df),
        fill=False, edgecolor='#762A83', lw=2.0
    ))
    
    return ax
```

### 8.2 ROC 曲线组合图（多诊断区分能力）

```python
from sklearn.metrics import roc_curve, auc

def plot_multi_roc(y_true_dict, y_score_dict, ax=None,
                   colors=None, title='Diagnostic Discrimination'):
    """
    绘制多分类 ROC 曲线
    y_true_dict: {'MDD': array, 'SCZ': array, ...}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(mm2inch(89), mm2inch(89)))
    
    if colors is None:
        colors = list(NATURE_2024.values())[:len(y_true_dict)]
    
    for (name, y_true), color in zip(y_true_dict.items(), colors):
        y_score = y_score_dict[name]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f'{name} (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    
    # 对角参考线
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6, zorder=1)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=7)
    ax.set_ylabel('True Positive Rate', fontsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.legend(loc='lower right', fontsize=6, frameon=False)
    
    despine(ax)
    return ax
```

### 8.3 特异性悖论可视化（Effect Size Heatmap）

```python
def plot_specificity_paradox(effect_df, ax=None):
    """
    可视化"特异性悖论"：各语言特征在不同诊断中的效应量差异
    effect_df: shape (n_features, n_diagnoses)，值为 Cohen's d 或 η²
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(mm2inch(183), mm2inch(80)))
    
    # 气泡图：大小=效应量绝对值，颜色=方向（正/负）
    n_features, n_diag = effect_df.shape
    
    for i, feature in enumerate(effect_df.index):
        for j, diag in enumerate(effect_df.columns):
            d = effect_df.loc[feature, diag]
            size = np.abs(d) * 200  # 气泡大小比例
            color = '#D6604D' if d > 0 else '#2166AC'
            alpha = min(0.9, 0.3 + np.abs(d) * 0.6)
            
            ax.scatter(j, i, s=size, c=color,
                      alpha=alpha, zorder=3,
                      edgecolors='white', linewidths=0.4)
            
            # 标注效应量数值（仅显著）
            if np.abs(d) > 0.2:
                ax.text(j, i, f'{d:.2f}', ha='center', va='center',
                       fontsize=4.5, color='white', fontweight='bold')
    
    ax.set_xticks(range(n_diag))
    ax.set_xticklabels(effect_df.columns, fontsize=6, rotation=30)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(effect_df.index, fontsize=6)
    
    ax.set_xlabel('Diagnosis', fontsize=7)
    ax.set_ylabel('Language Feature', fontsize=7)
    ax.set_title("Specificity Paradox: Cross-Diagnostic Effect Size Profile",
                fontsize=8, fontweight='bold')
    
    # 图例
    for d_val, label in [(0.2,'small'), (0.5,'medium'), (0.8,'large')]:
        ax.scatter([], [], s=d_val*200, c='gray', alpha=0.7,
                  label=f"|d| = {d_val} ({label})")
    ax.legend(loc='upper right', fontsize=6, frameon=True,
             framealpha=0.9, edgecolor='#E0E0E0')
    
    despine(ax)
    return ax
```

### 8.4 机器学习特征重要性图

```python
def plot_feature_importance(importance_df, ax=None, top_n=20,
                             model_name='XGBoost'):
    """
    水平条形图展示特征重要性（SHAP 值或 Gini）
    importance_df: columns=['feature', 'importance', 'ci_lower', 'ci_upper']
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(mm2inch(89), mm2inch(100)))
    
    df = importance_df.nlargest(top_n, 'importance').iloc[::-1]
    
    y_pos = range(len(df))
    
    # 误差棒
    xerr = [df['importance'] - df['ci_lower'],
            df['ci_upper'] - df['importance']]
    
    bars = ax.barh(y_pos, df['importance'],
                   xerr=xerr,
                   color='#4393C3',
                   edgecolor='white',
                   linewidth=0.5,
                   error_kw={'elinewidth': 0.8,
                             'capsize': 2.5,
                             'ecolor': '#636363'},
                   height=0.7,
                   alpha=0.85)
    
    # 特征名
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize=5.5)
    ax.set_xlabel(f'Feature Importance ({model_name})', fontsize=7)
    ax.set_title(f'Top {top_n} Predictive Features', fontsize=8,
                fontweight='bold')
    
    # 零线
    ax.axvline(0, color='black', lw=0.8)
    
    despine(ax)
    ax.invert_yaxis()
    return ax
```

### 8.5 跨诊断 UMAP / t-SNE 嵌入图

```python
def plot_embedding(embeddings, labels, ax=None,
                  method='UMAP', label_map=None, colors=None):
    """
    2D 嵌入散点图（UMAP / t-SNE）
    embeddings: shape (n_samples, 2)
    labels: array of group labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(mm2inch(89), mm2inch(89)))
    
    unique_labels = np.unique(labels)
    if colors is None:
        colors = WONG_8[:len(unique_labels)]
    if label_map is None:
        label_map = {l: l for l in unique_labels}
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=color, label=label_map[label],
            s=8, alpha=0.5,
            edgecolors='none',
            rasterized=True  # 大量点时加速渲染
        )
    
    ax.set_xlabel(f'{method}-1', fontsize=7)
    ax.set_ylabel(f'{method}-2', fontsize=7)
    ax.set_title(f'{method} of Social Text Features', fontsize=8,
                fontweight='bold')
    ax.legend(markerscale=2, fontsize=6, frameon=False,
             bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 隐藏刻度（嵌入图通常无单位意义）
    ax.set_xticks([])
    ax.set_yticks([])
    despine(ax, left=True, bottom=True)
    return ax
```

---

## 九、完整全局样式表

```python
# ============================================================
# nature_style.py
# 完整的 Nature/Science 风格 matplotlib 样式配置
# ============================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def apply_nature_style():
    """应用 Nature 期刊风格全局配置"""
    
    mpl.rcParams.update({
        # ── 字体 ──────────────────────────────────────────────
        'font.family':              'sans-serif',
        'font.sans-serif':          ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size':                7,
        'font.weight':              'regular',
        
        # ── 字号体系 ──────────────────────────────────────────
        'axes.titlesize':           8,
        'axes.labelsize':           7,
        'xtick.labelsize':          6,
        'ytick.labelsize':          6,
        'legend.fontsize':          7,
        'figure.titlesize':         9,
        
        # ── 图形 ──────────────────────────────────────────────
        'figure.dpi':               300,
        'figure.facecolor':         'white',
        'figure.edgecolor':         'none',
        'figure.autolayout':        False,  # 手动控制布局
        
        # ── 轴 ────────────────────────────────────────────────
        'axes.linewidth':           0.8,
        'axes.spines.top':          False,
        'axes.spines.right':        False,
        'axes.facecolor':           'white',
        'axes.labelpad':            4,
        'axes.titlepad':            6,
        'axes.titleweight':         'bold',
        
        # ── 刻度 ──────────────────────────────────────────────
        'xtick.major.width':        0.8,
        'ytick.major.width':        0.8,
        'xtick.major.size':         3.0,
        'ytick.major.size':         3.0,
        'xtick.minor.size':         1.5,
        'ytick.minor.size':         1.5,
        'xtick.direction':          'out',
        'ytick.direction':          'out',
        'xtick.major.pad':          3,
        'ytick.major.pad':          3,
        
        # ── 网格 ──────────────────────────────────────────────
        'axes.grid':                False,   # 默认关闭，按需开启
        'grid.color':               '#E0E0E0',
        'grid.linewidth':           0.5,
        'grid.linestyle':           '--',
        'grid.alpha':               0.7,
        'axes.axisbelow':           True,
        
        # ── 线条 ──────────────────────────────────────────────
        'lines.linewidth':          1.5,
        'lines.markersize':         4.0,
        'lines.markeredgewidth':    0.5,
        'errorbar.capsize':         3,
        
        # ── 填充/补丁 ─────────────────────────────────────────
        'patch.linewidth':          0.8,
        'patch.edgecolor':          'auto',
        
        # ── 图例 ──────────────────────────────────────────────
        'legend.frameon':           False,
        'legend.handlelength':      1.5,
        'legend.handleheight':      0.7,
        'legend.handletextpad':     0.5,
        'legend.borderaxespad':     0.5,
        'legend.columnspacing':     1.0,
        
        # ── 保存 ──────────────────────────────────────────────
        'savefig.dpi':              300,
        'savefig.bbox':             'tight',
        'savefig.pad_inches':       0.05,
        'savefig.facecolor':        'white',
        'savefig.edgecolor':        'none',
        'savefig.format':           'pdf',     # 矢量优先
        
        # ── PDF 设置 ──────────────────────────────────────────
        'pdf.fonttype':             42,        # TrueType 字体嵌入
        'ps.fonttype':              42,
        'svg.fonttype':             'none',    # SVG 保留可编辑文字
        
        # ── 颜色循环 ──────────────────────────────────────────
        'axes.prop_cycle': mpl.cycler('color', [
            '#0072B2', '#D55E00', '#009E73', '#E69F00',
            '#56B4E9', '#CC79A7', '#F0E442', '#000000'
        ]),
    })
    
    print("✓ Nature 风格已应用")

# 使用
apply_nature_style()
```

---

## 十、多面板组合图规范

### 10.1 面板布局与标签

```python
def add_panel_label(ax, label, x=-0.15, y=1.05):
    """添加 Nature 风格面板标签 (A, B, C...)"""
    ax.text(x, y, label,
            transform=ax.transAxes,
            fontsize=9, fontweight='bold',
            va='top', ha='left',
            fontfamily='Arial')

# 示例：3 面板图
fig = plt.figure(figsize=(mm2inch(183), mm2inch(120)))

# 使用 GridSpec 精控间距
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig,
              hspace=0.45,    # 行间距
              wspace=0.40,    # 列间距
              left=0.08,      # 左边距
              right=0.96,
              top=0.94,
              bottom=0.10)

ax_a = fig.add_subplot(gs[0, :2])   # 跨两列
ax_b = fig.add_subplot(gs[0, 2])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[1, 2])

for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], 'ABCDE'):
    add_panel_label(ax, label)
    despine(ax)
```

### 10.2 图例统一管理

```python
def unified_legend(fig, axes, ncol=4, y=-0.02):
    """
    在多面板图底部放置统一图例
    """
    handles, labels = [], []
    seen = set()
    
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
                ax.get_legend().remove() if ax.get_legend() else None
    
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, y),
               ncol=ncol,
               frameon=False,
               fontsize=7,
               handlelength=1.5)
```

---

## 十一、输出规范

### 11.1 文件格式建议

| 格式 | 用途 | DPI |
|------|------|-----|
| **PDF** | 主投稿文件（矢量）| — |
| **TIFF** | 高分辨率位图备用 | 300 dpi（彩图）/ 600 dpi（线图）|
| **EPS** | 部分期刊要求 | — |
| **SVG** | 后期 Illustrator 编辑 | — |
| **PNG** | 预览/汇报用 | 150–300 dpi |

### 11.2 输出代码

```python
def save_figure(fig, filename, formats=('pdf', 'tiff'), dpi=300):
    """Nature/Science 标准输出"""
    for fmt in formats:
        path = f"{filename}.{fmt}"
        kwargs = dict(dpi=dpi, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
        if fmt == 'tiff':
            kwargs['compression'] = 'tiff_lzw'
        if fmt == 'pdf':
            kwargs['backend'] = 'pdf'
        fig.savefig(path, format=fmt, **kwargs)
        print(f"✓ Saved: {path}")

# 调用
save_figure(fig, 'Figure_1_p_factor', formats=['pdf', 'tiff', 'svg'])
```

---

## 十二、计算精神病学特定图表 checklist

在提交图片前，请逐项核查：

**基本规范**
- [ ] 字体为 Arial，无宋体/黑体
- [ ] 最小字号 ≥ 6pt（300 DPI 下）
- [ ] 上轴、右轴已移除（despine）
- [ ] 颜色通过色盲测试（Sim Daltonism / Coblis）
- [ ] 图例文字完整且无缩写歧义

**统计标注**
- [ ] 所有统计检验已标注检验方法（t-test / Mann-Whitney U 等）
- [ ] p 值格式统一（< 0.001 or 精确值）
- [ ] 效应量已报告（Cohen's d / η² / AUC）
- [ ] 样本量 n 已在图内或图注标注
- [ ] 多重比较校正方法已说明（FDR / Bonferroni）

**p 因子相关**
- [ ] 因子载荷矩阵已标注旋转方法（Promax / Varimax）
- [ ] 模型拟合指标已在图注列出（CFI / RMSEA / SRMR）
- [ ] 特异性悖论效应量比较已进行组间检验

**机器学习**
- [ ] 交叉验证方案已标注（5-fold / LOOCV）
- [ ] SHAP 值单位轴标已说明
- [ ] 训练/测试集划分比例已标注

---

## 十三、推荐工具链

| 工具 | 用途 |
|------|------|
| `matplotlib` + `seaborn` | 主力绘图库 |
| `plotly` | 交互图（汇报用）|
| `scipy.stats` | 统计检验 |
| `pingouin` | 完整统计报告（效应量等）|
| `shap` | SHAP 特征重要性 |
| `umap-learn` | UMAP 嵌入 |
| `Adobe Illustrator` / `Inkscape` | PDF 矢量后处理 |
| `Color Oracle` | 色盲模拟测试 |
| `Coblis` (web) | 在线色盲检查 |

---

*本文档适用于 Nature、Science、Nature Medicine、Nature Human Behaviour、JAMA Psychiatry、Molecular Psychiatry 等期刊投稿图片规范。请以目标期刊官方 Author Guidelines 为最终标准。*
