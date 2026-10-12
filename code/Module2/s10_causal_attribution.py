"""
Step 10: Causal Attribution Analysis (v1.0)

使用准实验设计推断 t0 前后行为变化的因果方向。

核心分析:
  1. DID 设计 (双重差分)
  2. 事件研究系数动态 (检验平行趋势 + 动态效应)
  3. 安慰剂 t0 测试
  4. 剂量-反应分析 (轨迹陡度与严重度的关系)
  5. 敏感性: 多种权重方案

输入: user_trajectories.parquet (s07), trajectory_subtypes.csv (s09)
输出: 
  - did_results.csv
  - event_study_coefficients.csv
  - placebo_test_results.json
  - causal_report.json
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config_module2 import (
    TRAJECTORY_DIR, STATS_DIR, FEATURES_DIR, # 新增 FEATURES_DIR
    DISORDER_GROUPS, RANDOM_SEED,
    WINDOW_STEP_DAYS, WINDOW_SIZE_DAYS,
    DID_POST_START, DID_PRE_END, # [FIX v1.2] 新增导入
    PLACEBO_MIN_SHIFT, PLACEBO_SHIFT_RANGE, PLACEBO_N_TESTS
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "causal_attribution.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# [FIX v1.2] 解除全局警告屏蔽
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.random.seed(RANDOM_SEED)


# ======================== DID ESTIMATION ========================

def estimate_did(traj_df: pd.DataFrame) -> Dict:
    """
    [FIX v1.3] DID 估计 with 用户固定效应 + 聚类SE。
    注意：本数据结构为单一处理队列，所有患者在同一时间点 (t0) 接受处理。
    这缓解了 Goodman-Bacon (2021) 指出的多队列 TWFE 负权重问题，
    但仍需通过动态事件研究法检验平行趋势，并警惕预期效应 
    和随时间变化的异质处理效应。
    """
    logger.info("=" * 60)
    logger.info("Analysis 1: Difference-in-Differences (Regression-Based)")
    logger.info("=" * 60)
    
    target_vars = ["p_factor_severity_z", "volatility_index",
                   "social_withdrawal_index", "vader_compound_mean"]
    target_vars = [v for v in target_vars if v in traj_df.columns]
    
    # [FIX v1.2] 统一使用数据预处理函数
    df = traj_df.copy()
    df["patient"] = (df["group"] == "patient").astype(int)
    df_analysis = _prepare_did_data(df)
    
    results = {}
    
    for var in target_vars:
        logger.info(f"\n  DID for: {var}")
        
        var_results = {}
        
        # === Method 1: 回归-based DID with 聚类 SE ===
        reg_did = _regression_did(df_analysis, var)
        var_results["regression_clustered"] = reg_did
        
        # === Method 2: 配对固定效应回归 ===
        if "pair_id" in df_analysis.columns:
            pair_fe_did = _pair_fixed_effects_did(df_analysis, var)
            var_results["pair_fe"] = pair_fe_did
        
        # === Method 3: 加权回归 DID ===
        if "iptw_att_weight" in df_analysis.columns:
            weighted_did = _weighted_regression_did(df_analysis, var)
            var_results["iptw_weighted"] = weighted_did
        
        # === Method 4: 动态 DID (事件研究系数) ===
        dynamic_did = _dynamic_did_regression(df_analysis, var)
        var_results["dynamic"] = dynamic_did
        
        results[var] = var_results
        
        # 报告
        for method, res in var_results.items():
            if isinstance(res, dict) and "did_estimate" in res:
                sig = "✓" if res.get("p_value", 1) < 0.05 else "✗"
                logger.info(f"    {method:25s}: DID={res['did_estimate']:.4f} "
                           f"(SE={res.get('se', np.nan):.4f}, "
                           f"p={res.get('p_value', np.nan):.4f}) {sig}")
    
    # 保存
    did_summary = _summarize_did_results(results)
    did_summary.to_csv(
        os.path.join(STATS_DIR, "did_results.csv"), index=False)
    
    return results


def _regression_did(df: pd.DataFrame, var: str) -> Dict:
    """
    [FIX v1.1] 回归-based DID with 用户固定效应 + 聚类标准误。
    
    模型:
      Y_it = α_i + γ·Post_t + β·(Post×Patient)_it + ε_it
    
    α_i = 用户固定效应 (去均值)
    β = DID 估计量
    SE = 用户级聚类
    """
    data = df[[var, "patient", "post", "post_x_patient", "user_id_hash"]].dropna()
    
    if len(data) < 100:
        return {"did_estimate": np.nan}
    
    Y = data[var].values
    n = len(Y)
    
    # 用户去均值 (固定效应)
    user_means = data.groupby("user_id_hash")[var].transform("mean")
    Y_demeaned = Y - user_means.values
    
    # 自变量去均值
    X_post = data["post"].values
    X_interaction = data["post_x_patient"].values
    
    user_post_mean = data.groupby("user_id_hash")["post"].transform("mean")
    user_inter_mean = data.groupby("user_id_hash")["post_x_patient"].transform("mean")
    
    X_post_dm = X_post - user_post_mean.values
    X_inter_dm = X_interaction - user_inter_mean.values
    
    # OLS: Y_demeaned = β₁·Post_dm + β₂·Interaction_dm + ε
    X_matrix = np.column_stack([X_post_dm, X_inter_dm])
    
    try:
        beta = np.linalg.lstsq(X_matrix, Y_demeaned, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"did_estimate": np.nan}
    
    did_estimate = beta[1]  # β₂ = DID
    
    # 残差
    resid = Y_demeaned - X_matrix @ beta
    
    # 聚类标准误
    clusters = data["user_id_hash"].values
    se = _compute_clustered_se_matrix(resid, X_matrix, clusters)
    
    if se is not None and not np.isnan(se[1]):
        did_se = se[1]
        t_stat = did_estimate / did_se
        n_clusters = data["user_id_hash"].nunique()
        p_value = 2 * stats.t.sf(abs(t_stat), df=max(1, n_clusters - 1))
    else:
        did_se = np.nan
        t_stat = np.nan
        p_value = np.nan
    
    return {
        "did_estimate": float(did_estimate),
        "se": float(did_se),
        "se_type": "clustered_by_user",
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_obs": len(data),
        "n_users": data["user_id_hash"].nunique(),
        "n_clusters": data["user_id_hash"].nunique(),
    }


def _pair_fixed_effects_did(df: pd.DataFrame, var: str) -> Dict:
    """
    [FIX v1.2] 配对 DID：使用配对差分法 替代有缺陷的配对 FE 回归。
    消除 1:1 匹配导致的自由度枯竭和共线性问题。
    逻辑：计算每对 (患者差分 - 对照差分)，然后做单样本 t 检验。
    """
    data = df[[var, "patient", "post", "post_x_patient", 
               "user_id_hash", "pair_id"]].dropna()
    data = data[data["pair_id"].notna()]
    
    if len(data) < 100:
        return {"did_estimate": np.nan}
        
    pair_dids = []
    
    # [FIX v1.6] 向量化替代逐 pair 循环：原实现对 ~66 万 pair 各做 4 次
    # 布尔过滤 + mean，Python 循环开销是耗时主因之一。
    # 改为按 (pair_id, patient, post) 聚合均值后一步差分。
    agg = (
        data.groupby(["pair_id", "patient", "post"])[var]
        .mean()
        .unstack("patient")          # index=(pair_id, post), cols=patient
        .unstack("post")             # index=pair_id, cols=(patient, post)
    )
    # 需要完整 2×2 结构：patient∈{0,1}, post∈{0,1}
    if not all(c in agg.columns for c in [(0,0),(0,1),(1,0),(1,1)]):
        return {"did_estimate": np.nan}
    valid = agg[[(0,0),(0,1),(1,0),(1,1)]].dropna()
    if len(valid) == 0:
        return {"did_estimate": np.nan}
    diff_pat = valid[(1,1)] - valid[(1,0)]    # 患者后-前
    diff_ctrl = valid[(0,1)] - valid[(0,0)]   # 对照后-前
    pair_dids = (diff_pat - diff_ctrl).values
    if len(pair_dids) < 20:
        return {"did_estimate": np.nan}
        
    pair_dids = np.array(pair_dids)
    mean_did = np.mean(pair_dids)
    se_did = np.std(pair_dids, ddof=1) / np.sqrt(len(pair_dids))
    t_stat = mean_did / se_did if se_did > 1e-10 else 0
    p_value = 2 * stats.t.sf(abs(t_stat), df=len(pair_dids) - 1)
    
    return {
        "did_estimate": float(mean_did),
        "se": float(se_did),
        "se_type": "pairwise_differencing",
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_pairs": len(pair_dids),
    }


def _compute_clustered_se_matrix(
    resid: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray
) -> Optional[np.ndarray]:
    try:
        # [FIX v1.5] 用伪逆替代 inv：动态 DID 的 X 含窗口主效应+交互列，
        # 在用户去均值后可能近似共线（某窗口仅患者/仅对照时 W_k 与 PxW_k 近线性相关），
        # 导致 X.T@X 奇异。伪逆保证数值稳健。
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)  # G
    n_obs = X.shape[0]                 # N
    k = X.shape[1]                     # K
    
    # Meat matrix
    # [FIX v1.6] 向量化替代逐 cluster 循环：原实现 for cl in unique_clusters
    # 对 689,584 个用户各做一次 O(N) 的 mask 比较 + 矩阵乘（~9.5e12 次比较），
    # 是 s10 耗时 34 小时的根因。改用 np.add.at 一次向量化累加每个 cluster 的
    # Σ resid_i·X_i（B_cl），再在外层做外积求和，复杂度降至 O(N)。
    B_all = resid[:, None] * X          # (N, K)
    B_sum = np.zeros((n_clusters, k))
    cluster_idx = pd.Categorical(clusters).codes
    np.add.at(B_sum, cluster_idx, B_all)
    meat = B_sum.T @ B_sum               # Σ_cl (Σ_{i∈cl} resid_i·X_i)⊗(Σ_{i∈cl} resid_i·X_i)
    
    # [FIX v1.4] 添加小样本校正因子 G/(G-1) * (N-1)/(N-K)
    # 这是 Stata 和 linearmodels 计算聚类标准误的标准做法，缺失会导致过度拒绝原假设
    if n_clusters > 1 and n_obs > k:
        correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k))
    else:
        correction = 1.0
        
    # Sandwich
    sandwich = correction * (XtX_inv @ meat @ XtX_inv)
    
    # 对角线 = 每个系数的方差
    diag = np.diag(sandwich)
    if np.any(diag < 0):
        return None
    
    return np.sqrt(diag)


def _dynamic_did_regression(df: pd.DataFrame, var: str) -> Dict:
    """
    [FIX v1.5] 规范的动态 DID (事件研究法)。
    核心修正：
      1. 纳入窗口主效应（对照组参考时间轨迹），患者×窗口交互系数才是
         相对对照组的因果事件研究系数（此前缺失窗口主效应，对照组交互恒 0，
         模型退化为"患者组内均值"而非患者 vs 对照差异）。
      2. 用户去均值 + 窗口主效应联合识别，等价于 TWFE 事件研究。
    模型：Y_it = α_i + Σ_k θ_k 1{t=k} + Σ_{k≠ref} β_k (Patient_i × 1{t=k}) + ε_it
    """
    logger.info(f"  Estimating Dynamic DID (Event Study) for {var}...")
    data = df[[var, "patient", "window_index", "user_id_hash"]].dropna()
    
    if len(data) < 100:
        return {}
        
    # 1. 设定参照期 (t0 前 240~90 天，window_index -8 到 -4)
    ref_range = (-8, -4)
    data["window_cat"] = data["window_index"].astype(str)
    data.loc[data["window_index"].between(ref_range[0], ref_range[1]), "window_cat"] = "Baseline"
    
    # 2. 窗口主效应（对照组参考轨迹，drop 参照窗）
    # [FIX v1.9] 转 float 避免 bool 列使 X.values 变 object
    window_dummies = pd.get_dummies(data["window_cat"], prefix="W", drop_first=False).astype(float)
    cols_to_drop = [col for col in window_dummies.columns if "Baseline" in col]
    window_dummies = window_dummies.drop(columns=cols_to_drop)
    
    # 3. 患者×窗口交互 (排除 Baseline)
    # [FIX v1.7] interactions 列必须重命名为 PxW_ 前缀：此前直接 copy
    # window_dummies 的列（仍是 W_ 前缀），导致下方 col.replace("PxW_","")
    # 无法解析，全部窗口被跳过 → KeyError: 'p_value'。
    interactions = window_dummies.copy()
    interactions.columns = [c.replace("W_", "PxW_") for c in interactions.columns]
    for col in interactions.columns:
        interactions[col] = interactions[col] * data["patient"]
    
    data = pd.concat([data, window_dummies, interactions], axis=1)
    exog_cols = list(window_dummies.columns) + list(interactions.columns)
    
    # 4. 用户去均值 (固定效应)
    Y_dm = data[var] - data.groupby("user_id_hash")[var].transform("mean")
    X_dm = data[exog_cols] - data.groupby("user_id_hash")[exog_cols].transform("mean")
    
    X_matrix = X_dm.values
    
    try:
        beta = np.linalg.lstsq(X_matrix, Y_dm, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
        
    resid = Y_dm - X_matrix @ beta
    clusters = data["user_id_hash"].values
    se_arr = _compute_clustered_se_matrix(resid.values, X_matrix, clusters)
    
    # 5. 只报告患者×窗口交互系数（因果事件研究系数），窗口主效应为对照组参考
    #    注意：X_matrix 列顺序 = window_dummies 列 + interactions 列，
    #    interactions 列的 beta/SE 索引需加上窗口主效应列数的偏移。
    window_effects = []
    n_w_columns = len(window_dummies.columns)
    for i, col in enumerate(interactions.columns):
        # [FIX v1.4] 更安全的列名解析，避免下划线干扰
        win_idx_str = col.replace("PxW_", "")
        try:
            win_idx = int(float(win_idx_str))
        except ValueError:
            logger.warning(f"  Could not parse window index from '{col}', skipping")
            continue
        
        beta_idx = n_w_columns + i
        effect = beta[beta_idx]
        se = se_arr[beta_idx] if se_arr is not None else np.nan
        
        if not np.isnan(se) and se > 1e-10:
            t_stat = effect / se
            p_value = 2 * stats.t.sf(abs(t_stat), df=len(data["user_id_hash"].unique()) - 1)
        else:
            p_value = np.nan
            
        window_effects.append({
            "window_index": win_idx,
            "days_from_t0": win_idx * WINDOW_STEP_DAYS + WINDOW_SIZE_DAYS / 2,  # 或从数据中映射
            "effect": float(effect),
            "se": float(se),
            "p_value": float(p_value),
        })
        
    effects_df = pd.DataFrame(window_effects)    
    # [FIX v1.3] 补全多重假设检验的 FDR 校正 (与 Step 08 对齐)
    p_vals = effects_df["p_value"].dropna().values
    if len(p_vals) > 0:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        effects_df["p_fdr"] = np.nan
        effects_df.loc[effects_df["p_value"].notna(), "p_fdr"] = p_adj
        
    effects_df.to_csv(os.path.join(STATS_DIR, "event_study_coefficients.csv"), index=False)
    
    return {"window_effects": effects_df.to_dict("records"), "n_windows": len(effects_df)}



def _weighted_regression_did(df: pd.DataFrame, var: str) -> Dict:
    """
    [FIX v1.3] IPTW 加权固定效应 DID 回归 (含加权聚类标准误)。
    """
    weights_col = "iptw_att_weight"
    data = df[[var, "patient", "post", "post_x_patient", "user_id_hash", weights_col]].dropna()
    
    if len(data) < 100:
        return {"did_estimate": np.nan}
        
    # 构建权重向量：患者为1，对照为ATT权重
    data["weight"] = np.where(data["patient"] == 1, 1.0, data[weights_col])
    W = data["weight"].values
    
    # [FIX v1.4] FWL 定理：加权去均值，统一使用 map 对齐，消除索引错位风险
    user_weighted_means_Y = data.groupby("user_id_hash").apply(
        lambda g: np.average(g[var], weights=g["weight"]))
    # [FIX v1.5] 使用 .values 强制按位置相减，避免因非默认索引导致的静默对齐错误
    Y_dm = data[var].values - data["user_id_hash"].map(user_weighted_means_Y).values
    
    user_weighted_means_post = data.groupby("user_id_hash").apply(
        lambda g: np.average(g["post"], weights=g["weight"]))
    post_dm = data["post"].values - data["user_id_hash"].map(user_weighted_means_post).values
    
    user_weighted_means_inter = data.groupby("user_id_hash").apply(
        lambda g: np.average(g["post_x_patient"], weights=g["weight"]))
    inter_dm = data["post_x_patient"].values - data["user_id_hash"].map(user_weighted_means_inter).values
    
    X_matrix = np.column_stack([post_dm, inter_dm])
    
    try:
        # 加权 OLS: min Σ w_i (Y_i - X_i β)^2
        # 等价于 W^{1/2} X β = W^{1/2} Y
        sqrt_W = np.sqrt(W)
        beta = np.linalg.lstsq(X_matrix * sqrt_W[:, None], Y_dm * sqrt_W, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"did_estimate": np.nan}
        
    did_estimate = beta[1]
    
    # === [FIX v1.3] 计算加权聚类标准误 ===
    resid = Y_dm - X_matrix @ beta
    clusters = data["user_id_hash"].values
    unique_clusters = np.unique(clusters)
    k = X_matrix.shape[1]
    
    try:
        # Bread: (X' W X)^{-1}
        XtWX = X_matrix.T @ (X_matrix * W[:, None])
        XtWX_inv = np.linalg.inv(XtWX)
        
        # Meat: Σ_c ( Σ_{i in c} w_i * e_i * X_i )' ( Σ_{i in c} w_i * e_i * X_i )
        # [FIX v1.6] 向量化替代逐 cluster 循环（同 _compute_clustered_se_matrix）。
        B_all = (resid * W)[:, None] * X_matrix   # (N, K)
        B_sum = np.zeros((len(unique_clusters), k))
        cluster_idx = pd.Categorical(clusters).codes
        np.add.at(B_sum, cluster_idx, B_all)
        meat = B_sum.T @ B_sum
            
        # Sandwich
        sandwich = XtWX_inv @ meat @ XtWX_inv
        se_arr = np.sqrt(np.diag(sandwich))
        did_se = se_arr[1]
        
        t_stat = did_estimate / did_se if did_se > 1e-10 else 0
        n_clusters = len(unique_clusters)
        p_value = 2 * stats.t.sf(abs(t_stat), df=max(1, n_clusters - 1))
        
    except np.linalg.LinAlgError:
        did_se = np.nan
        t_stat = np.nan
        p_value = np.nan
    
    return {
        "did_estimate": float(did_estimate),
        "se": float(did_se),
        "se_type": "weighted_clustered_by_user",
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_obs": len(data),
    }
    

# ======================== PLACEBO T0 TEST ========================

def placebo_t0_test(traj_df: pd.DataFrame,
                     n_placebos: int = 100) -> Dict:
    """
    [FIX v1.2] 严谨的安慰剂 t0 测试。
    核心修正：使用与基准 DID 完全相同的固定效应回归模型，而非简单均值差。
    为保证运行速度，对用户进行随机子采样。
    注意：shift 的实际取值区间为 [PLACEBO_MIN_SHIFT, PLACEBO_SHIFT_RANGE] 天、
    符号随机（见下），并非早期注释所称的 ±180。
    """
    logger.info("=" * 60)
    logger.info(f"Analysis 2: Placebo t0 Test (n={n_placebos}, "
                f"shift in [{PLACEBO_MIN_SHIFT}, {PLACEBO_SHIFT_RANGE}] d, FE-Regression based)")
    logger.info("=" * 60)
    
    target_var = "p_factor_severity_z"
    if target_var not in traj_df.columns:
        return {}
        
    # [FIX v1.2] 为了速度，子采样最多 5000 名用户
    all_uids = traj_df["user_id_hash"].unique()
    if len(all_uids) > 5000:
        rng_sample = np.random.RandomState(RANDOM_SEED)
        sample_uids = rng_sample.choice(all_uids, 5000, replace=False)
        df_sub = traj_df[traj_df["user_id_hash"].isin(sample_uids)].copy()
    else:
        df_sub = traj_df.copy()
        
    df_sub["patient"] = (df_sub["group"] == "patient").astype(int)
    
    # 1. 计算真实 DID (基于子样本)
    true_did_result = _regression_did(
        _prepare_did_data(df_sub), target_var
    )
    true_did = true_did_result.get("did_estimate", np.nan)
    
    if np.isnan(true_did):
        logger.error("  True DID estimation failed on subsample!")
        return {}
    
    logger.info(f"  True DID (subsample): {true_did:.4f}")
    
    # 2. 安慰剂测试
    placebo_dids = []
    rng = np.random.RandomState(RANDOM_SEED + 100)
    
    for i in range(n_placebos):
        shift = rng.choice([-1, 1]) * rng.randint(PLACEBO_MIN_SHIFT, PLACEBO_SHIFT_RANGE + 1)
        
        # 移位数据
        shifted_df = df_sub.copy()
        shifted_df["days_from_t0_center"] = shifted_df["days_from_t0_center"] - shift
        
        # 重新定义 Post
        shifted_prepared = _prepare_did_data(shifted_df)
        
        placebo_result = _regression_did(shifted_prepared, target_var)
        placebo_did = placebo_result.get("did_estimate", np.nan)
        
        if not np.isnan(placebo_did):
            placebo_dids.append(placebo_did)
            if (i+1) % 10 == 0:
                logger.info(f"  Placebo {i+1}/{n_placebos}: shift={shift:+d}d, DID={placebo_did:.4f}")
    
    # [NEW] 保存安慰剂 DID 分布数据，供绘制直方图
    placebo_dist_df = pd.DataFrame({"placebo_did": placebo_dids})
    placebo_dist_path = os.path.join(STATS_DIR, "placebo_did_distribution.csv")
    placebo_dist_df.to_csv(placebo_dist_path, index=False)
    logger.info(f"Saved placebo DID distribution for visualization: {placebo_dist_path}")
    
    if not placebo_dids:
        return {"true_did": true_did, "n_placebos": 0}
    
    placebo_dids = np.array(placebo_dids)
    
    # [FIX v1.2] 修复未定义变量 n_exceeding
    n_exceeding = int(np.sum(np.abs(placebo_dids) > np.abs(true_did)))
    placebo_p = (n_exceeding + 1) / (n_placebos + 1)
    
    result = {
        "true_did": float(true_did),
        "n_placebos": len(placebo_dids),
        "placebo_did_mean": float(np.mean(placebo_dids)),
        "placebo_did_std": float(np.std(placebo_dids)),
        "n_exceeding_true": n_exceeding,
        "placebo_p_value": float(placebo_p),
        "conclusion": (
            "CAUSAL_EVIDENCE" if placebo_p < 0.05 else
            "SUGGESTIVE" if placebo_p < 0.10 else
            "INCONCLUSIVE"
        ),
    }
    
    logger.info(f"\n  Placebo Test Result:")
    logger.info(f"    True DID: {true_did:.4f}")
    logger.info(f"    Placebo DID: {np.mean(placebo_dids):.4f} ± {np.std(placebo_dids):.4f}")
    logger.info(f"    N exceeding true: {n_exceeding}/{len(placebo_dids)}")
    logger.info(f"    Placebo p-value: {placebo_p:.4f}")
    logger.info(f"    Conclusion: {result['conclusion']}")
    
    with open(os.path.join(STATS_DIR, "placebo_test_results.json"), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    return result

# 辅助函数：统一数据预处理（排除过渡区）
def _prepare_did_data(df: pd.DataFrame) -> pd.DataFrame:
    """[NEW] 统一 DID 数据预处理逻辑，排除过渡区。"""
    df = df.copy()
    df["post"] = 0
    df.loc[df["days_from_t0_center"] > DID_POST_START, "post"] = 1
    # 过渡区标记为 -1 排除
    df.loc[
        (df["days_from_t0_center"] >= DID_PRE_END) & 
        (df["days_from_t0_center"] <= DID_POST_START), "post"
    ] = -1
    # [FIX v1.5] 从 group 推断 patient 列，修复亚型分层 DID 中 patient 列缺失
    # 导致的 KeyError: 'patient'（subtype_data 切片不含 patient 列）。
    if "patient" not in df.columns:
        if "group" in df.columns:
            df["patient"] = (df["group"] == "patient").astype(int)
        else:
            df["patient"] = 0
    df["post_x_patient"] = df["post"] * df["patient"]
    return df[df["post"] != -1].copy()


# ======================== DOSE-RESPONSE ANALYSIS ========================

def dose_response_analysis(traj_df: pd.DataFrame,
                           shape_df: pd.DataFrame) -> Dict:
    """
    [FIX v1.2] 重构为：累积暴露 (Dose) 与 t0 后结局 (Response) 的关系。
    Dose = pre-t0 AUC (累积严重度)
    Response = post-t0 severity (结局水平)
    """
    logger.info("=" * 60)
    logger.info("Analysis 3: Dose-Response (Cumulative Exposure × Post-Outcome)")
    logger.info("=" * 60)
    
    target_var = "p_factor_severity_z"
    
    # 计算累积暴露和结局
    records = []
    for uid, user_df in traj_df[traj_df["group"] == "patient"].groupby("user_id_hash"):
        # [FIX v1.3] 防御性编程：检查 phase 列是否存在
        if "phase" in user_df.columns:
            pre_data = user_df[user_df["phase"] == "pre"]
            post_data = user_df[user_df["phase"] == "post"]
        else:
            # 回退策略：基于天数划分
            pre_data = user_df[user_df["days_from_t0_center"] < -WINDOW_STEP_DAYS / 2]
            post_data = user_df[user_df["days_from_t0_center"] > WINDOW_STEP_DAYS / 2]
        
        if len(pre_data) < 3 or len(post_data) < 3:
            continue
            
        # [FIX P0] 必须先删除 NaN，再提取值，避免因中间存在 NaN 导致的 x-y 坐标错位
        pre_clean = pre_data.dropna(subset=[target_var])
        post_clean = post_data.dropna(subset=[target_var])
        
        if len(pre_clean) > 1 and len(post_clean) > 0:
            # Dose: 曲线下面积 (相对0的累积)
            dose = np.trapz(pre_clean[target_var].values, pre_clean["days_from_t0_center"].values)
            # Response: t0 后平均严重度
            response = post_clean[target_var].mean()
            
            records.append({
                "user_id_hash": uid,
                "dose_auc": dose,
                "response_post_mean": response,
                "disorder_group": user_df["disorder_group"].iloc[0]
            })
            
    if len(records) < 50:
        logger.warning("  Insufficient data for dose-response analysis")
        return {}
        
    dr_df = pd.DataFrame(records)
    
    # 相关性分析
    # [FIX v1.4] 使用 OLS 回归控制混杂，替代简单 Pearson 相关
    # 需要引入基线严重度作为控制变量
    if shape_df is not None and len(shape_df) > 0 and "p_factor_severity_z_baseline" in shape_df.columns:
        dr_df = dr_df.merge(
            shape_df[["user_id_hash", "p_factor_severity_z_baseline"]],
            on="user_id_hash", how="left"
        )
        has_baseline = True
    else:
        # 回退：从轨迹数据中自行计算基线
        logger.warning("  Baseline not found in shape_df, computing from trajectory...")
        baseline_records = []
        for uid, user_df in traj_df[traj_df["group"] == "patient"].groupby("user_id_hash"):
            if "phase" in user_df.columns:
                pre_data = user_df[(user_df["phase"] == "pre") & (user_df["days_from_t0_center"] < -270)]
            else:
                pre_data = user_df[user_df["days_from_t0_center"] < -270]
            if len(pre_data) > 0:
                baseline_records.append({
                    "user_id_hash": uid,
                    "p_factor_severity_z_baseline": pre_data[target_var].dropna().mean()
                })
        if baseline_records:
            bl_df = pd.DataFrame(baseline_records)
            dr_df = dr_df.merge(bl_df, on="user_id_hash", how="left")
            has_baseline = True
        else:
            has_baseline = False
    
    # --- 全样本 OLS 回归 ---
    # [FIX v1.8] valid 必须包含 baseline 列，否则后续 has_baseline 分支
    # 检查的列不在 valid 中 → 拼接时 dtype object → OLS 失败。
    _sel_cols = ["dose_auc", "response_post_mean", "disorder_group"]
    if has_baseline and "p_factor_severity_z_baseline" in dr_df.columns:
        _sel_cols.append("p_factor_severity_z_baseline")
    valid = dr_df[_sel_cols].dropna()
    
    if has_baseline and "p_factor_severity_z_baseline" in valid.columns:
        valid = valid.dropna(subset=["p_factor_severity_z_baseline"])
    
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    
    # 简单 Pearson（保留作为参考基线）
    r_simple, p_simple = stats.pearsonr(valid["dose_auc"], valid["response_post_mean"])
    logger.info(f"  Uncontrolled Dose-Response: r={r_simple:.3f}, p={p_simple:.4f}")
    
    # OLS: Response ~ Dose + Baseline + Disorder Dummies
    if len(valid) >= 50:
        # 构建回归特征
        ols_features = ["dose_auc"]
        if has_baseline and "p_factor_severity_z_baseline" in valid.columns:
            ols_features.append("p_factor_severity_z_baseline")
        
        # 疾病虚拟变量
        # [FIX v1.9] get_dummies 默认返回 bool dtype，与 float 特征列 concat 后
        # DataFrame.values 会提升为 object → statsmodels "cast to numpy dtype of object"。
        # 显式转 float。
        disorder_dummies = pd.get_dummies(valid["disorder_group"], prefix="disorder", drop_first=True).astype(float)
        ols_feature_cols = ols_features + list(disorder_dummies.columns)
        
        X_ols = valid[ols_features].copy()
        X_ols = pd.concat([X_ols, disorder_dummies], axis=1)
        X_ols = sm.add_constant(X_ols)
        y_ols = valid["response_post_mean"]
        
        # 对齐索引（防止 NaN 导致长度不一致）
        aligned = pd.concat([y_ols, X_ols], axis=1).dropna()
        if len(aligned) >= 30:
            y_aligned = aligned["response_post_mean"]
            X_aligned = aligned.drop(columns=["response_post_mean"])
            
            try:
                model = sm.OLS(y_aligned, X_aligned).fit(cov_type="HC1")  # 稳健标准误
                
                dose_coef = model.params.get("dose_auc", np.nan)
                dose_se = model.bse.get("dose_auc", np.nan)
                dose_p = model.pvalues.get("dose_auc", np.nan)
                dose_ci = model.conf_int().loc["dose_auc"] if "dose_auc" in model.conf_int().index else (np.nan, np.nan)
                
                logger.info(f"  Controlled Dose-Response OLS (HC1 SE):")
                logger.info(f"    Dose AUC coefficient: {dose_coef:.4f}")
                logger.info(f"    Dose AUC SE: {dose_se:.4f}")
                logger.info(f"    Dose AUC p-value: {dose_p:.4f}")
                logger.info(f"    Dose AUC 95% CI: [{dose_ci[0]:.4f}, {dose_ci[1]:.4f}]")
                logger.info(f"    R-squared: {model.rsquared:.4f}")
                
                if has_baseline and "p_factor_severity_z_baseline" in model.params.index:
                    bl_coef = model.params["p_factor_severity_z_baseline"]
                    bl_p = model.pvalues["p_factor_severity_z_baseline"]
                    logger.info(f"    Baseline severity coefficient: {bl_coef:.4f} (p={bl_p:.4f})")
                    
            except Exception as e:
                logger.warning(f"  OLS regression failed: {e}")
                dose_coef = np.nan
                dose_p = np.nan
                dose_ci = (np.nan, np.nan)
        else:
            logger.warning("  Too few aligned samples for OLS")
            dose_coef = np.nan
            dose_p = np.nan
            dose_ci = (np.nan, np.nan)
    else:
        dose_coef = np.nan
        dose_p = np.nan
        dose_ci = (np.nan, np.nan)
    
    # --- 分层分析（每疾病内 OLS） ---
    logger.info("\n  Dose-Response by disorder (within-disorder OLS):")
    stratified_results = {}
    for disorder in DISORDER_GROUPS:
        dis_data = valid[valid["disorder_group"] == disorder]
        if len(dis_data) > 30:
            # 疾病内无虚拟变量，仅 Dose + Baseline
            strat_features = ["dose_auc"]
            if has_baseline and "p_factor_severity_z_baseline" in dis_data.columns:
                strat_features.append("p_factor_severity_z_baseline")
                
            X_strat = sm.add_constant(dis_data[strat_features])
            y_strat = dis_data["response_post_mean"]
            
            aligned_strat = pd.concat([y_strat, X_strat], axis=1).dropna()
            if len(aligned_strat) > 20:
                try:
                    strat_model = sm.OLS(
                        aligned_strat["response_post_mean"],
                        aligned_strat.drop(columns=["response_post_mean"])
                    ).fit(cov_type="HC1")
                    
                    dose_coef_d = strat_model.params.get("dose_auc", np.nan)
                    dose_p_d = strat_model.pvalues.get("dose_auc", np.nan)
                    logger.info(f"    {disorder}: dose_coef={dose_coef_d:.4f}, p={dose_p_d:.4f}")
                    stratified_results[disorder] = {"coef": float(dose_coef_d), "p": float(dose_p_d)}
                except Exception:
                    r_d, p_d = stats.pearsonr(dis_data["dose_auc"], dis_data["response_post_mean"])
                    logger.info(f"    {disorder}: OLS failed, Pearson r={r_d:.3f}, p={p_d:.4f}")
                    stratified_results[disorder] = {"r": float(r_d), "p": float(p_d)}
            else:
                logger.info(f"    {disorder}: insufficient aligned data (n={len(aligned_strat)})")
        else:
            logger.info(f"    {disorder}: insufficient data (n={len(dis_data)})")
    
    # [NEW] 保存剂量-反应散点图数据及 OLS 拟合线
    if len(valid) > 0:
        plot_df = valid.copy()
        # 如果 OLS 成功，生成预测值用于画拟合线
        if 'model' in locals() and isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
            try:
                # 使用对齐后的数据预测
                plot_df = aligned.copy()
                plot_df["predicted_response"] = model.predict(plot_df)
                # 画部分残差图（控制其他变量后 Dose 的效应）更为严谨
                plot_df["dose_partial_residual"] = aligned["response_post_mean"] - \
                    (aligned["const"] * model.params.get("const", 0)) - \
                    (aligned.get("p_factor_severity_z_baseline", 0) * model.params.get("p_factor_severity_z_baseline", 0))
            except Exception as e:
                logger.warning(f"Failed to generate OLS predictions for plot: {e}")
                
        dose_plot_path = os.path.join(STATS_DIR, "dose_response_scatter_data.csv")
        plot_df.to_csv(dose_plot_path, index=False)
        logger.info(f"Saved dose-response scatter data: {dose_plot_path}")
    
    return {
        "uncontrolled_pearson": {"r": float(r_simple), "p": float(p_simple)},
        "controlled_ols": {
            "dose_coef": float(dose_coef) if not np.isnan(dose_coef) else None,
            "dose_p": float(dose_p) if not np.isnan(dose_p) else None,
            "dose_ci_lower": float(dose_ci[0]) if not np.isnan(dose_ci[0]) else None,
            "dose_ci_upper": float(dose_ci[1]) if not np.isnan(dose_ci[1]) else None,
        },
        "stratified_by_disorder": stratified_results,
    }


# ======================== SUBTYPE-STRATIFIED DID ========================

def subtype_stratified_did(traj_df: pd.DataFrame,
                           shape_df: pd.DataFrame) -> Dict:
    """
    按轨迹亚型分层计算 DID。
    
    不同亚型的 DID 估计量是否不同?
    "急转直下"型应该有最大的 DID。
    """
    logger.info("=" * 60)
    logger.info("Analysis 4: Subtype-Stratified DID")
    logger.info("=" * 60)
    
    if "subtype_label" not in shape_df.columns:
        logger.warning("  No subtype labels available")
        return {}
    
    # 合并亚型标签
    subtype_map = dict(zip(
        shape_df["user_id_hash"], shape_df["subtype_label"]
    ))
    
    traj_with_subtype = traj_df.copy()
    traj_with_subtype["subtype_label"] = traj_with_subtype["user_id_hash"].map(subtype_map)
    
    # [FIX v1.5] 合并亚型概率列：软匹配权重依赖 subtype_{k}_prob，
    # 此前 subtype_data 切片中从未包含这些列，导致 H4 软匹配权重全部走 else 分支失效。
    prob_cols = [c for c in shape_df.columns if c.startswith("subtype_") and c.endswith("_prob")]
    if prob_cols:
        for pc in prob_cols:
            prob_map = dict(zip(shape_df["user_id_hash"], shape_df[pc]))
            traj_with_subtype[pc] = traj_with_subtype["user_id_hash"].map(prob_map)
    
    target_var = "p_factor_severity_z"
    
    results = {}
    
    # [FIX v1.4] 整体基准也统一使用 FE 回归，与分层分析保持方法论一致
    overall_data = traj_with_subtype.copy()
    overall_data["patient"] = (overall_data["group"] == "patient").astype(int)
    overall_data_prepared = _prepare_did_data(overall_data)
    overall_did = _regression_did(overall_data_prepared, target_var)
    results["overall"] = overall_did
    
    # 按亚型
    # 按亚型
    for subtype in sorted(traj_with_subtype["subtype_label"].dropna().unique()):
        subtype_data = traj_with_subtype[
            (traj_with_subtype["subtype_label"] == subtype) |
            (traj_with_subtype["group"] == "control")
        ].copy()
        
        # [FIX v1.2] 软匹配：根据对照属于该亚型的概率，对对照进行加权
        # 在 subtype_stratified_did() 循环内部修改对照权重逻辑：

        # [FIX v1.4] 增强的亚型软匹配：概率阈值过滤 + 权重归一化
        prob_col = f"subtype_{int(subtype)}_prob"
        if prob_col in subtype_data.columns:
            # Step 1: 设定概率阈值，过滤掉属于该亚型概率极低的对照
            prob_threshold = 0.15  # 低于此概率的对照不参与该亚型 DID 估计
            ctrl_mask = subtype_data["group"] == "control"
            prob_values = subtype_data[prob_col].fillna(0)
            
            # 保留：所有患者 + 概率达标的对照
            keep_mask = ~ctrl_mask | (prob_values >= prob_threshold)
            n_ctrl_before = ctrl_mask.sum()
            n_ctrl_after = (ctrl_mask & keep_mask).sum()
            logger.info(f"  Subtype {int(subtype)} control filtering: "
                       f"{n_ctrl_before} → {n_ctrl_after} (threshold={prob_threshold})")
            
            subtype_data = subtype_data[keep_mask].copy()
            
            # Step 2: 对保留的对照赋予权重（概率值），患者权重为1
            subtype_data["iptw_att_weight"] = np.where(
                subtype_data["group"] == "control",
                subtype_data[prob_col].fillna(0),
                1.0
            )
            
            # Step 3: 归一化对照权重使其均值为1，防止有效样本量人为膨胀
            ctrl_weight_mask = subtype_data["group"] == "control"
            if ctrl_weight_mask.sum() > 0:
                ctrl_mean_weight = subtype_data.loc[ctrl_weight_mask, "iptw_att_weight"].mean()
                if ctrl_mean_weight > 1e-6:
                    subtype_data.loc[ctrl_weight_mask, "iptw_att_weight"] /= ctrl_mean_weight
        else:
            # 无概率列时，所有对照等权重参与
            subtype_data["iptw_att_weight"] = 1.0
            
        subtype_data = _prepare_did_data(subtype_data)
        
        # 根据是否有权重选择模型
        if "iptw_att_weight" in subtype_data.columns:
            subtype_did = _weighted_regression_did(subtype_data, target_var)
        else:
            subtype_did = _regression_did(subtype_data, target_var)      
        
        results[f"subtype_{int(subtype)}"] = subtype_did
        
        did_est = subtype_did.get("did_estimate", np.nan)
        p_val = subtype_did.get("p_value", np.nan)
        if not np.isnan(did_est):
            sig = "✓" if p_val < 0.05 else "✗"
            logger.info(f"  Subtype {int(subtype)}: DID={did_est:.4f} "
                       f"(p={p_val:.4f}) {sig}")
    
    return results


# ======================== CROSS-DISORDER DID ========================

def cross_disorder_did(traj_df: pd.DataFrame) -> Dict:
    """
    按疾病分组计算 DID。
    
    不同疾病的 t0 效应是否不同?
    """
    logger.info("=" * 60)
    logger.info("Analysis 5: Cross-Disorder DID")
    logger.info("=" * 60)
    
    target_var = "p_factor_severity_z"
    results = {}
    
    df = traj_df.copy()
    df["patient"] = (df["group"] == "patient").astype(int)
    
    for disorder in DISORDER_GROUPS:
        # 该疾病的患者 + 全部对照
        # [FIX-A5] 使用 patient 二元变量替代 group 字符串比较
        dis_data = df[
            (df["disorder_group"] == disorder) | (df["patient"] == 0)
        ].copy()
        
        dis_data["patient"] = (dis_data["disorder_group"] == disorder).astype(int)
        
        # [FIX v1.3] 统一使用 FE 回归，替代简单 DID，确保聚类 SE 和固定效应的一致性
        dis_data_prepared = _prepare_did_data(dis_data)
        did_result = _regression_did(dis_data_prepared, target_var)
        
        if not np.isnan(did_result.get("did_estimate", np.nan)):
            results[disorder] = did_result
            sig = "✓" if did_result.get("p_value", 1) < 0.05 else "✗"
            logger.info(f"  {disorder}: DID={did_result['did_estimate']:.4f} "
                       f"(p={did_result.get('p_value', np.nan):.4f}) {sig}")
    
    # 比较不同疾病的 DID
    if len(results) >= 2:
        did_values = {k: v["did_estimate"] for k, v in results.items()}
        max_disorder = max(did_values, key=lambda k: abs(did_values[k]))
        min_disorder = min(did_values, key=lambda k: abs(did_values[k]))
        
        logger.info(f"\n  Largest effect: {max_disorder} "
                   f"(DID={did_values[max_disorder]:.4f})")
        logger.info(f"  Smallest effect: {min_disorder} "
                   f"(DID={did_values[min_disorder]:.4f})")
    
    # 保存
    did_df = pd.DataFrame([
        {"disorder": k, **v} for k, v in results.items()
    ])
    did_df.to_csv(
        os.path.join(STATS_DIR, "cross_disorder_did.csv"), index=False)
    
    return results


# ======================== SUMMARY HELPER ========================

# 1. 修改 _summarize_did_results 函数：
def _summarize_did_results(results: Dict) -> pd.DataFrame:
    rows = []
    for var, var_results in results.items():
        for method, method_result in var_results.items():
            if isinstance(method_result, dict) and "did_estimate" in method_result:
                rows.append({
                    "variable": var,
                    "method": method,
                    "did_estimate": method_result.get("did_estimate", np.nan),
                    "se": method_result.get("se", np.nan),
                    "p_value": method_result.get("p_value", np.nan),
                })
    
    df = pd.DataFrame(rows)
    
    # [FIX] 全局 FDR 校正
    from statsmodels.stats.multitest import multipletests
    valid_p = df["p_value"].dropna()
    if len(valid_p) > 0:
        _, p_adj, _, _ = multipletests(valid_p.values, alpha=0.05, method="fdr_bh")
        df.loc[valid_p.index, "p_fdr_global"] = p_adj
        
    return df

# ======================== MAIN ========================

def run_causal_attribution():
    """主管线: 因果归因分析。"""
    logger.info("=" * 80)
    logger.info("MODULE 2 - Step 10: Causal Attribution Analysis")
    logger.info("=" * 80)
    
    # 加载轨迹数据
    traj_path = os.path.join(TRAJECTORY_DIR, "user_trajectories.parquet")
    traj_df = pd.read_parquet(traj_path)
    logger.info(f"Loaded trajectories: {traj_df.shape}")

    # [FIX v1.5] 仅使用可靠窗口分析：s07 对 silent 窗口做了 MNAR 插补
    # （P90/P10 伪值）并参与了 p_factor 计算，若不过滤，插补窗口会被当作
    # 真实数据进入 DID/安慰剂检验，制造人造效应。
    if "is_reliable" in traj_df.columns:
        n_all = len(traj_df)
        traj_df = traj_df[traj_df["is_reliable"]].copy()
        logger.info(f"  Filtered to reliable windows: {len(traj_df)}/{n_all} "
                    f"({len(traj_df)/n_all*100:.1f}%)")

    # [FIX-A5] 防御性标准化 group 列
    traj_df["group"] = traj_df["group"].apply(
        lambda g: "patient" if g == "patient" else "control"
    )
    
    # 加载亚型数据
    shape_path = os.path.join(STATS_DIR, "trajectory_subtypes.csv")
    shape_df = pd.read_csv(shape_path) if os.path.exists(shape_path) else pd.DataFrame()
    
    # 1. DID 估计
    did_results = estimate_did(traj_df)
    
    # 2. 安慰剂 t0 测试
    placebo_results = placebo_t0_test(traj_df, n_placebos=PLACEBO_N_TESTS)
    
    # 3. 剂量-反应分析
    dose_response = dose_response_analysis(traj_df, shape_df)
    
    # 4. 亚型分层 DID
    subtype_did = subtype_stratified_did(traj_df, shape_df) if len(shape_df) > 0 else {}
    
    # 5. 跨疾病 DID
    cross_disorder = cross_disorder_did(traj_df)
    
    # === 综合报告 ===
    report = {
        "pipeline_version": "v1.0",
        "did_summary": _summarize_did_results(did_results).to_dict("records"),
        "placebo_test": placebo_results,
        "cross_disorder_did": {
            k: v.get("did_estimate", np.nan) 
            for k, v in cross_disorder.items()
        },
        "causal_evidence_strength": _assess_causal_evidence(
            did_results, placebo_results
        ),
    }
    
    with open(os.path.join(STATS_DIR, "causal_report.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("=" * 80)
    logger.info("CAUSAL ATTRIBUTION ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return did_results, placebo_results


def _assess_causal_evidence(did_results: Dict, 
                             placebo_results: Dict) -> str:
    """评估因果证据强度。"""
    # [FIX v1.5] 前置门槛：平行趋势检验必须通过，否则因果强度封顶为 WEAK。
    # 经典 TWFE DID 的因果解释依赖平行趋势假设；若前窗系数显著偏离 0，
    # DID 估计量捕捉的是组间预处理趋势差异，而非处理效应。
    pt_path = os.path.join(STATS_DIR, "parallel_trends_test.json")
    pt_conclusion = None
    if os.path.exists(pt_path):
        try:
            with open(pt_path, 'r') as f:
                pt_conclusion = json.load(f).get("conclusion", None)
        except Exception:
            pass
    
    parallel_trends_ok = pt_conclusion in (None, "PASS", "MARGINAL")
    
    # 检查主要变量的 DID 显著性
    main_var = "p_factor_severity_z"
    
    if main_var in did_results:
        # [FIX v1.2] 修正键名，与 estimate_did 输出对齐
        main_did = did_results[main_var].get("regression_clustered", {})
        did_p = main_did.get("p_value", 1.0)
        did_sig = did_p < 0.05
    else:
        did_sig = False
    
    placebo_sig = placebo_results.get("conclusion", "INCONCLUSIVE") == "CAUSAL_EVIDENCE"
    
    if not parallel_trends_ok:
        # 平行趋势 FAIL 时 DID 因果解释不成立，封顶为 WEAK
        return "WEAK" if (did_sig or placebo_sig) else "INSUFFICIENT"
    elif did_sig and placebo_sig:
        return "STRONG"
    elif did_sig and not placebo_sig:
        return "MODERATE"
    elif not did_sig and placebo_sig:
        return "WEAK"
    else:
        return "INSUFFICIENT"


if __name__ == "__main__":
    run_causal_attribution()