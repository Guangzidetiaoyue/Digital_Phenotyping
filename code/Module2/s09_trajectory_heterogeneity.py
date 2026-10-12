"""
Step 09: Trajectory Heterogeneity Analysis (v1.2)

识别轨迹亚型: "急转直下" vs "长期侵蚀" vs "波动回弹"

核心方法:
  1. 基于特征的模型聚类
     注意: 本模块采用两步法——先提取具有临床可解释性的轨迹形状特征，
     再使用高斯混合模型 进行聚类。
     这与传统纵向建模中的生长混合模型 概念不同，
     但功能等价于基于函数型数据的聚类，且特征可解释性更强。
  2. 亚型预测因子分析 (基于 Permutation Importance)
  3. 亚型与 Module 1 表型的映射

输入: user_trajectories.parquet (s07), changepoint_analysis.csv (s08, 可选)
输出: 
  - trajectory_subtypes.csv (每用户一个亚型标签)
  - subtype_profiles.json
  - subtype_predictors.csv
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config_module2 import (
    TRAJECTORY_DIR, STATS_DIR,
    DISORDER_GROUPS, RANDOM_SEED,
    WINDOW_STEP_DAYS,
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "trajectory_heterogeneity.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# [FIX v1.3] 解除全局警告屏蔽，暴露 GMM 收敛失败、奇异矩阵等关键警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# 保留 RuntimeWarning 和 ConvergenceWarning，以便审计模型是否真正收敛
np.random.seed(RANDOM_SEED)


# ======================== TRAJECTORY SHAPE FEATURES ========================

def extract_trajectory_shape_features(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX v1.1] 增强的轨迹形状特征提取。
    
    新增:
      - 疾病进程速度 (基线到t0的时间尺度)
      - 恢复特征 (t0后是否恢复、恢复速度)
      - 交互特征 (pre-slope × post-slope)
      - 极值特征 (最大/最小severity、达到时间)
      - 曲线下面积 (累积severity)
      - 转折锐度 (斜率变化的角度)
    """
    logger.info("Extracting trajectory shape features (enhanced)...")
    
    target_vars = ["p_factor_severity_z"]
    target_vars = [v for v in target_vars if v in traj_df.columns]
    
    # [FIX v1.2] 桥接 Step 08：加载拐点检测结果，作为聚类的核心特征之一
    cp_path = os.path.join(STATS_DIR, "changepoint_analysis.csv")
    user_cp_counts = {}
    if os.path.exists(cp_path):
        cp_df = pd.read_csv(cp_path)
        # 统计每个用户 t0 前的拐点数量
        pre_cps = cp_df[cp_df["is_pre_t0"] & (cp_df["group"] == "patient")]
        user_cp_counts = pre_cps.groupby("user_id_hash").size().to_dict()
        logger.info(f"  Loaded Step 08 changepoints: {len(user_cp_counts)} users with pre-t0 CPs")
    
    shape_records = []
    
    for uid, user_df in tqdm(traj_df.groupby("user_id_hash"),
                              desc="Shape features"):
        sorted_df = user_df.sort_values("window_index")
        
        record = {
            "user_id_hash": uid,
            "group": sorted_df["group"].iloc[0],
            "disorder_group": sorted_df["disorder_group"].iloc[0],
        }
        
        # [FIX v1.2] 引入 Step 08 的拐点数量
        record["n_pre_t0_changepoints"] = user_cp_counts.get(uid, 0)
        
        for var in target_vars:
            series = sorted_df[var].values
            days = sorted_df["days_from_t0_center"].values
            valid = ~np.isnan(series)
            if "is_reliable" in sorted_df.columns:
                valid &= sorted_df["is_reliable"].values
            
            if valid.sum() < 6:
                continue
            
            s = series[valid]
            d = days[valid]
            
            # ============================================
            # 原有特征 (保留)
            # ============================================
            
            # [FIX v1.2] 严格基线定义，与 s07/s08 对齐，排除前驱期污染
            baseline_mask = (d < -270) & (d > -360)
            baseline_vals = s[baseline_mask]
            record[f"{var}_baseline"] = np.mean(baseline_vals) if len(baseline_vals) > 0 else np.nan
            
            # t0 后水平
            post_mask = (d > 30) & (d < 150)
            post_vals = s[post_mask]
            record[f"{var}_post_level"] = np.mean(post_vals) if len(post_vals) > 0 else np.nan
            
            # 阶跃高度
            if not np.isnan(record.get(f"{var}_baseline", np.nan)) and \
               not np.isnan(record.get(f"{var}_post_level", np.nan)):
                record[f"{var}_step_height"] = (
                    record[f"{var}_post_level"] - record[f"{var}_baseline"]
                )
            
            # t0 前斜率
            pre_mask = d < -WINDOW_STEP_DAYS / 2
            pre_s, pre_d = s[pre_mask], d[pre_mask]
            if len(pre_s) >= 3:
                slope_pre, intercept_pre, r_pre, p_pre, se_pre = stats.linregress(pre_d, pre_s)
                record[f"{var}_slope_pre"] = slope_pre
                record[f"{var}_r_squared_pre"] = r_pre ** 2
            else:
                record[f"{var}_slope_pre"] = np.nan
                record[f"{var}_r_squared_pre"] = np.nan
            
            # t0 后斜率
            post_mask_full = d > WINDOW_STEP_DAYS / 2
            post_s, post_d = s[post_mask_full], d[post_mask_full]
            if len(post_s) >= 3:
                slope_post, _, r_post, _, _ = stats.linregress(post_d, post_s)
                record[f"{var}_slope_post"] = slope_post
                record[f"{var}_r_squared_post"] = r_post ** 2
            else:
                record[f"{var}_slope_post"] = np.nan
                record[f"{var}_r_squared_post"] = np.nan
            
            # 加速度
            if not np.isnan(record.get(f"{var}_slope_pre", np.nan)) and \
               not np.isnan(record.get(f"{var}_slope_post", np.nan)):
                record[f"{var}_acceleration"] = (
                    record[f"{var}_slope_post"] - record[f"{var}_slope_pre"]
                )
            
            # 波动性
            if len(pre_s) >= 3:
                pre_predicted = slope_pre * pre_d + intercept_pre
                record[f"{var}_volatility_pre"] = np.std(pre_s - pre_predicted)
            
            if len(post_s) >= 3:
                slope_post_val = record.get(f"{var}_slope_post", 0)
                intercept_post = post_s[0] - slope_post_val * post_d[0] \
                    if not np.isnan(slope_post_val) else np.mean(post_s)
                post_predicted = slope_post_val * post_d + intercept_post
                record[f"{var}_volatility_post"] = np.std(post_s - post_predicted)
            
            # ============================================
            # [NEW] 增强特征
            # ============================================
            
            # --- 1. 疾病进程速度 ---
            # 从基线到 t0 的时间 (基于严重度从基线上升1个z-score的点)
            if not np.isnan(record.get(f"{var}_baseline", np.nan)):
                baseline_val = record[f"{var}_baseline"]
                # 找到首次超过 baseline + 1.0 的时间
                above_threshold = s > (baseline_val + 1.0)
                if above_threshold.any():
                    first_above_idx = np.argmax(above_threshold)
                    record[f"{var}_onset_speed_days"] = d[first_above_idx]
                else:
                    record[f"{var}_onset_speed_days"] = np.nan
            
            # --- 2. 恢复特征 ---
            if len(post_s) >= 3:
                # 是否恢复到基线 (t0 后 180~360 天)
                late_post_mask = (d > 150) & (d < 360)
                late_post_vals = s[late_post_mask]
                
                if len(late_post_vals) > 0 and \
                   not np.isnan(record.get(f"{var}_baseline", np.nan)):
                    late_post_mean = np.mean(late_post_vals)
                    baseline_val = record[f"{var}_baseline"]
                    
                    # 恢复比例 (1.0 = 完全恢复)
                    step = record.get(f"{var}_step_height", 0)
                    if abs(step) > 0.1:
                        recovery_pct = 1.0 - (late_post_mean - baseline_val) / step
                        record[f"{var}_recovery_pct"] = float(np.clip(recovery_pct, -1, 2))
                    else:
                        record[f"{var}_recovery_pct"] = 1.0  # 无需恢复
                    
                    # 是否恢复到基线 ±0.5 z-score
                    record[f"{var}_recovered_to_baseline"] = (
                        abs(late_post_mean - baseline_val) < 0.5
                    )
                
                # 恢复速度 (t0 后斜率的绝对值, 负 = 恢复中)
                record[f"{var}_recovery_slope"] = -record.get(f"{var}_slope_post", np.nan)
            
            # --- 3. 交互特征 ---
            slope_pre = record.get(f"{var}_slope_pre", np.nan)
            slope_post = record.get(f"{var}_slope_post", np.nan)
            
            if not np.isnan(slope_pre) and not np.isnan(slope_post):
                # 斜率乘积 (负 = 转折, 正 = 延续)
                record[f"{var}_slope_product"] = slope_pre * slope_post
                
                # 转折锐度 (角度变化)
                angle_pre = np.arctan(slope_pre * 30)  # 乘以30转换为每30天
                angle_post = np.arctan(slope_post * 30)
                record[f"{var}_turning_angle"] = abs(angle_post - angle_pre)
            
            # --- 4. 极值特征 ---
            if len(s) > 0:
                # 最大严重度
                max_idx = np.argmax(s)
                record[f"{var}_max_severity"] = s[max_idx]
                record[f"{var}_max_severity_days"] = d[max_idx]
                
                # 最小严重度
                min_idx = np.argmin(s)
                record[f"{var}_min_severity"] = s[min_idx]
                record[f"{var}_min_severity_days"] = d[min_idx]
                
                # 极值范围
                record[f"{var}_severity_range"] = np.ptp(s)
            
            # --- 5. 曲线下面积 (AUC) ---
            if len(s) > 1:
                # t0 前 AUC (累积严重度)
                pre_all_mask = d < -WINDOW_STEP_DAYS / 2
                pre_all_s = s[pre_all_mask]
                pre_all_d = d[pre_all_mask]
                if len(pre_all_s) > 1:
                    record[f"{var}_auc_pre"] = np.trapz(pre_all_s, pre_all_d)
                
                # t0 后 AUC
                post_all_mask = d > WINDOW_STEP_DAYS / 2
                post_all_s = s[post_all_mask]
                post_all_d = d[post_all_mask]
                if len(post_all_s) > 1:
                    record[f"{var}_auc_post"] = np.trapz(post_all_s, post_all_d)
                
                # AUC 比值 (post / |pre|)
                auc_pre = record.get(f"{var}_auc_pre", np.nan)
                auc_post = record.get(f"{var}_auc_post", np.nan)
                if not np.isnan(auc_pre) and not np.isnan(auc_post) and abs(auc_pre) > 0.01:
                    record[f"{var}_auc_ratio"] = auc_post / abs(auc_pre)
            
            # --- 6. 方向变化次数 (保留，但增加权重) ---
            if len(s) >= 4:
                diffs = np.diff(s)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                record[f"{var}_direction_changes"] = sign_changes
                # 归一化方向变化率 (每单位时间)
                time_span = d[-1] - d[0]
                if time_span > 0:
                    record[f"{var}_direction_change_rate"] = sign_changes / (time_span / 30)
            
            # --- 7. t0 附近变化幅度 (保留) ---
            near_t0_mask = (d > -60) & (d < 60)
            near_t0_vals = s[near_t0_mask]
            if len(near_t0_vals) >= 2:
                record[f"{var}_near_t0_range"] = np.ptp(near_t0_vals)
        
        shape_records.append(record)
    
    shape_df = pd.DataFrame(shape_records)
    logger.info(f"  Shape features extracted: {shape_df.shape}")
    
    return shape_df


# ======================== SUBTYPE DISCOVERY ========================

def discover_trajectory_subtypes(shape_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    基于轨迹形状特征发现潜在亚型。
    
    方法:
      1. 选择关键形状特征
      2. 标准化
      3. GMM 聚类 (2-5类)
      4. BIC/轮廓系数选择最优类别数
      5. 解释亚型
    """
    logger.info("=" * 60)
    logger.info("Trajectory Subtype Discovery (GMM)")
    logger.info("=" * 60)
    
    # 仅患者
    patients = shape_df[shape_df["group"] == "patient"].copy()
    logger.info(f"  Patients for clustering: {len(patients)}")
    
    # [FIX v1.2] 剔除 direction_changes (受观测长度严重干扰)，改用 rate；剔除共线性的 acceleration
    shape_cols = [
        "p_factor_severity_z_slope_pre",
        "p_factor_severity_z_slope_post",
        "p_factor_severity_z_step_height",
        "p_factor_severity_z_volatility_pre",
        "p_factor_severity_z_volatility_post",
        "p_factor_severity_z_direction_change_rate", # 替换 direction_changes
        "p_factor_severity_z_turning_angle",         # 新增：更综合的转折指标
        "n_pre_t0_changepoints", # 新增
    ]
    shape_cols = [c for c in shape_cols if c in patients.columns]
    
    # [FIX v1.4] 基于 VIF 的共线性特征自动筛选
    # 先对有效样本做 VIF 检测，剔除 VIF > 10 的高共线性特征
    _temp_valid = patients[shape_cols].dropna()
    if len(_temp_valid) > len(shape_cols) + 10:  # 确保样本量足够计算 VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        _vif_dropped = []
        _remaining_cols = shape_cols.copy()
        
        while len(_remaining_cols) > 2:  # 至少保留2个特征
            try:
                _vif_data = _temp_valid[_remaining_cols]
                # 标准化后计算 VIF（避免量纲干扰）
                _vif_data_std = (_vif_data - _vif_data.mean()) / _vif_data.std()
                
                # [FIX P1] 必须添加常数项，否则 statsmodels 的 VIF 计算会因未中心化 R² 导致数值爆炸
                import statsmodels.api as sm
                _vif_data_std = sm.add_constant(_vif_data_std)
                
                _vif_values = [
                    variance_inflation_factor(_vif_data_std.values, i)
                    for i in range(1, _vif_data_std.shape[1]) # 跳过第0列的 const
                ]
                _max_vif_idx = np.argmax(_vif_values)
                _max_vif = _vif_values[_max_vif_idx]
                
                if _max_vif > 10.0:  # VIF 阈值
                    _dropped = _remaining_cols.pop(_max_vif_idx)
                    _vif_dropped.append((_dropped, _max_vif))
                    logger.info(f"  VIF Filter: Dropped '{_dropped}' (VIF={_max_vif:.1f})")
                else:
                    break  # 所有 VIF <= 10，停止
            except Exception:
                break
        
        if _vif_dropped:
            shape_cols = _remaining_cols
            logger.info(f"  VIF filtering: {len(shape_cols)} features retained "
                       f"({len(_vif_dropped)} dropped)")
            for name, vif_val in _vif_dropped:
                logger.info(f"    Dropped: {name} (VIF={vif_val:.1f})")
    else:
        logger.warning(f"  Too few samples for VIF filtering (n={len(_temp_valid)}), using all features")
    
    # 去除缺失
    valid_mask = patients[shape_cols].notna().all(axis=1)
    patients_valid = patients[valid_mask].copy()
    logger.info(f"  Patients with complete shape features: {len(patients_valid)}")
    
    if len(patients_valid) < 100:
        logger.error("  Too few patients for clustering!")
        return shape_df, {}
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(patients_valid[shape_cols].values)
    
    # === GMM 模型选择 ===
    bic_scores = {}
    silhouette_scores = {}
    
    # [FIX v1.4] 协方差类型自适应降级策略：full → tied → diag
    covariance_priority = ["full", "tied", "diag"]
    
    for k in range(2, 6):
        gmm = None
        for cov_type in covariance_priority:
            try:
                gmm = GaussianMixture(
                    n_components=k, 
                    covariance_type=cov_type,
                    reg_covar=1e-3,  # 加大正则化防止奇异矩阵
                    random_state=RANDOM_SEED,
                    n_init=5,
                    max_iter=200,
                )
                gmm.fit(X)
                logger.info(f"  k={k}: Successfully fitted with covariance_type='{cov_type}'")
                break  # 成功则跳出降级循环
            except ValueError as e:
                logger.warning(f"  k={k}: covariance_type='{cov_type}' failed ({str(e)[:60]}), trying next...")
                gmm = None
                continue
        
        if gmm is None:
            logger.error(f"  k={k}: All covariance types failed, skipping")
            continue
            
        labels = gmm.predict(X)
        
        bic_scores[k] = gmm.bic(X)
        
        if len(set(labels)) > 1:
            silhouette_scores[k] = silhouette_score(X, labels)
        else:
            silhouette_scores[k] = -1
        
        logger.info(f"  k={k}: BIC={bic_scores[k]:.0f}, "
                    f"Silhouette={silhouette_scores[k]:.3f}")
    
    # 选择最优 k (BIC 最小 + Silhouette 可接受)
    best_k = min(bic_scores, key=bic_scores.get)
    
    # 如果 Silhouette 太低, 尝试更少的类
    if silhouette_scores.get(best_k, 0) < 0.15 and best_k > 2:
        for k in sorted(bic_scores.keys()):
            if silhouette_scores.get(k, 0) >= 0.15:
                best_k = k
                break
    
    logger.info(f"  Selected k={best_k}")
    
    # [NEW] 保存 GMM 模型选择指标，供绘图使用
    model_selection_path = os.path.join(STATS_DIR, "gmm_model_selection.csv")
    ms_df = pd.DataFrame([
        {"n_components": k, "bic": bic_scores.get(k), "silhouette": silhouette_scores.get(k)}
        for k in range(2, 6)
    ])
    ms_df.to_csv(model_selection_path, index=False)
    logger.info(f"Saved GMM model selection metrics: {model_selection_path}")
    
    # === 最终聚类 ===
    # [FIX v1.4] 协方差类型自适应降级：full → tied → diag
    best_gmm = None
    for cov_type in covariance_priority:
        try:
            best_gmm = GaussianMixture(
                n_components=best_k,
                covariance_type=cov_type,
                reg_covar=1e-3,  # 加大正则化防止奇异矩阵
                random_state=RANDOM_SEED,
                n_init=10,
                max_iter=300,
            )
            best_gmm.fit(X)
            logger.info(f"  Final model: Successfully fitted with covariance_type='{cov_type}'")
            break
        except ValueError as e:
            logger.warning(f"  Final model: covariance_type='{cov_type}' failed, trying next...")
            best_gmm = None
            continue
    
    if best_gmm is None:
        logger.error("  All covariance types failed for final model!")
        return shape_df, {}
        
    labels = best_gmm.predict(X)
    probs = best_gmm.predict_proba(X)
    
    # === [FIX v1.3] 聚类稳定性验证：100 次 Bootstrap ARI ===
    logger.info("  Running Bootstrap Stability Validation (100 iterations, aligned ARI)...")
    from sklearn.metrics import adjusted_rand_score
    
    ari_scores = []
    n_bootstrap = 100
    
    for i in range(n_bootstrap):
        # 有放回抽样，生成与原数据集同等大小的 Bootstrap 样本
        bs_idx = np.random.choice(len(X), size=len(X), replace=True)
        X_bs = X[bs_idx]
        
        try:
            gmm_bs = GaussianMixture(
                n_components=best_k, covariance_type="diag", reg_covar=1e-4, 
                random_state=RANDOM_SEED + i, n_init=3, max_iter=200
            )
            gmm_bs.fit(X_bs)
            labels_bs = gmm_bs.predict(X_bs)
            
            # 获取基准模型在同一 Bootstrap 样本上的标签
            labels_ref_bs = labels[bs_idx]
            
            # [核心修复] 必须先对齐标签，再计算 ARI！
            aligned_labels_bs = _align_cluster_labels(labels_ref_bs, labels_bs, best_k)
            
            ari = adjusted_rand_score(labels_ref_bs, aligned_labels_bs)
            ari_scores.append(ari)
        except Exception:
            # 如果某次 Bootstrap 因奇异矩阵失败，跳过
            continue
            
    if len(ari_scores) > 10:
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        logger.info(f"  Bootstrap ARI: {mean_ari:.3f} ± {std_ari:.3f} (n={len(ari_scores)})")
        if mean_ari < 0.6:
            logger.warning("  WARNING: Low ARI indicates subtype structure may be unstable or arbitrary!")
    else:
        logger.warning("  Too few successful Bootstrap iterations for ARI calculation")
    
    # === [FIX v1.2] 标签对齐逻辑：防止标签翻转 ===
    # 约定：按照 step_height 的中位数降序排列，确保 Subtype 0 始终是"急性/高阶跃"簇
    # === [FIX v1.2] 标签对齐逻辑：防止标签翻转 ===
    temp_df = pd.DataFrame({"label": labels, "step": patients_valid["p_factor_severity_z_step_height"].values})
    median_step = temp_df.groupby("label")["step"].median()
    
    # [FIX v1.6] 增加次级排序键防止平局导致的不确定性
    # 次级键：簇大小 (n)，确保即使 step 相同，排序也唯一
    cluster_sizes = temp_df.groupby("label").size()
    # 创建一个 DataFrame 进行复合排序
    sort_df = pd.DataFrame({
        'step': median_step,
        'size': cluster_sizes
    }).sort_values(by=['step', 'size'], ascending=[False, False])
    
    sort_order = sort_df.index.tolist()
    label_map = {old: new for new, old in enumerate(sort_order)}
    labels = np.vectorize(label_map.get)(labels)
    
    logger.info(f"  Label alignment mapping (by step_height desc): {label_map}")
    
    
    patients_valid["subtype_label"] = labels
    # [FIX v1.7] 概率列必须按 label_map 重排，否则与重排后的 subtype_label 错位
    inv_label_map = {v: k for k, v in label_map.items()}
    for i in range(best_k):
        patients_valid[f"subtype_{i}_prob"] = probs[:, inv_label_map[i]]
    
    # === 亚型解释 ===
    subtype_interpretation = _interpret_subtypes(
        patients_valid, shape_cols, best_k
    )
    
    # 将标签合并回完整 shape_df
    subtype_map = dict(zip(
        patients_valid["user_id_hash"], 
        patients_valid["subtype_label"]
    ))
    shape_df["subtype_label"] = shape_df["user_id_hash"].map(subtype_map)
    
    for i in range(best_k):
        prob_map = dict(zip(
            patients_valid["user_id_hash"],
            patients_valid[f"subtype_{i}_prob"]
        ))
        shape_df[f"subtype_{i}_prob"] = shape_df["user_id_hash"].map(prob_map)
    
    # === 对照组亚型检验 ===
    _check_control_subtypes(shape_df, shape_cols, scaler, best_gmm, best_k, label_map)
    
    return shape_df, subtype_interpretation


def _interpret_subtypes(
    patients_df: pd.DataFrame, 
    shape_cols: List[str],
    n_subtypes: int
) -> Dict:
    """
    [FIX v1.2] 基于簇间相对比较的数据驱动命名，摒弃全局阈值。
    逻辑：计算各簇在核心指标上的中位数，找到该簇最极端的特征来命名。
    """
    interpretation = {}
    
    # 核心判别指标
    core_features = {
        "p_factor_severity_z_step_height": "step",
        "p_factor_severity_z_slope_pre": "slope_pre",
        "p_factor_severity_z_slope_post": "slope_post",
        "p_factor_severity_z_volatility_pre": "volatility",
        "p_factor_severity_z_turning_angle": "turning_angle",
        "p_factor_severity_z_recovery_pct": "recovery"
    }
    
    # 计算各簇的中位数特征矩阵
    subtype_medians = {}
    for subtype_id in range(n_subtypes):
        subtype_data = patients_df[patients_df["subtype_label"] == subtype_id]
        subtype_medians[subtype_id] = subtype_data[core_features.keys()].median().to_dict()

    for subtype_id in range(n_subtypes):
        subtype_data = patients_df[patients_df["subtype_label"] == subtype_id]
        n = len(subtype_data)
        medians = subtype_medians[subtype_id]
        
        disorder_dist = subtype_data["disorder_group"].value_counts().to_dict()
        
        # [核心逻辑] 相对比较：该簇的指标在所有簇中的排名
        ranks = {}
        for feat_key, feat_short in core_features.items():
            if feat_key in medians and not np.isnan(medians[feat_key]):
                # 绝对值排名（1 = 最极端）
                all_vals = [subtype_medians[j].get(feat_key, 0) for j in range(n_subtypes)]
                if feat_short in ["recovery"]: # 恢复率看原值
                    ranks[feat_short] = sorted(all_vals, reverse=True).index(medians[feat_key]) + 1
                else: # 其他看绝对值
                    ranks[feat_short] = sorted([abs(v) for v in all_vals], reverse=True).index(abs(medians[feat_key])) + 1
        
        # 基于排名命名 (优先级: step > slope > vol > recovery)
        name = "Mixed Pattern"
        description = ""
        
        # [FIX v1.5] 调整 if-elif 顺序：step==1 AND slope_pre==1 (Acute on Chronic)
        # 必须优先于 recovery==1 (Resilient/Recovery)，否则同时满足两条件的簇
        # 会被误命名为 Resilient/Recovery（本次运行 Subtype 0 即此情形）。
        if ranks.get("step") == 1 and ranks.get("slope_pre") == 1:
            name = "Acute on Chronic"
            description = f"Both high pre-t0 erosion AND sudden step deterioration at t0"
        elif ranks.get("step") == 1 and ranks.get("slope_pre") > 1:
            name = "Acute Crisis"
            description = f"Relatively flat pre-t0, but sharpest sudden deterioration at t0 (step={medians.get('p_factor_severity_z_step_height', 0):.2f}σ)"
        elif ranks.get("slope_pre") == 1 and ranks.get("step") > 1:
            name = "Gradual Erosion"
            description = f"Steepest progressive worsening before t0, without a sudden break (slope={medians.get('p_factor_severity_z_slope_pre', 0):.4f}/d)"
        elif ranks.get("volatility") == 1:
            name = "Fluctuating"
            description = f"Highest volatility and direction changes, no clear linear trend (vol={medians.get('p_factor_severity_z_volatility_pre', 0):.3f})"
        elif ranks.get("recovery") == 1:
            name = "Resilient/Recovery"
            description = f"Significant event at t0, but highest recovery rate post-t0 (recovery={medians.get('p_factor_severity_z_recovery_pct', 0):.1%})"
        else:
            name = "Stable/Low-grade"
            description = f"Least pronounced changes across all dimensions"
            
        interpretation[f"subtype_{subtype_id}"] = {
            "name": name,
            "description": description,
            "n_patients": n,
            "pct_of_total": n / len(patients_df) * 100,
            "feature_medians": {k: float(v) for k, v in medians.items()},
            "disorder_distribution": disorder_dist,
            "ranks_among_subtypes": ranks,
        }
        
        logger.info(f"  Subtype {subtype_id} ({name}): n={n} ({n/len(patients_df)*100:.1f}%)")
        logger.info(f"    Ranks: {ranks}")
    
    return interpretation

def _align_cluster_labels(labels_ref: np.ndarray, labels_target: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    [FIX v1.3] 使用匈牙利算法 解决聚类标签置换问题。
    在计算 ARI 或比较分布前，必须对齐标签，否则相同结构的簇会因标签不同被误判。
    """
    from scipy.optimize import linear_sum_assignment
    
    # 构建列联表 (混淆矩阵)
    contingency = np.zeros((n_clusters, n_clusters))
    for i in range(len(labels_ref)):
        contingency[labels_ref[i], labels_target[i]] += 1
        
    # 使用匈牙利算法找到最优匹配 (最大化一致性)
    row_ind, col_ind = linear_sum_assignment(contingency, maximize=True)
    
    # 构建映射字典 {旧标签: 新标签}
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_labels = np.array([mapping[l] for l in labels_target])
    
    return aligned_labels

def _check_control_subtypes(
    shape_df: pd.DataFrame,
    shape_cols: List[str],
    scaler: StandardScaler,
    gmm: GaussianMixture,
    n_subtypes: int,
    label_map: dict = None,
):
    """
    检验对照组是否也有类似的亚型结构。
    
    方法: 将对照的形状特征投影到患者的 GMM 模型上，
    看对照是否也会被分到类似的类别。
    """
    logger.info("  Checking control group subtype structure...")
    
    controls = shape_df[shape_df["group"] == "control"].copy()
    valid_mask = controls[shape_cols].notna().all(axis=1)
    controls_valid = controls[valid_mask]
    
    if len(controls_valid) < 50:
        logger.info("  Too few controls with complete features")
        return
    
    X_ctrl = scaler.transform(controls_valid[shape_cols].values)
    ctrl_labels = gmm.predict(X_ctrl)
    
    # [FIX v1.7] 对照组必须应用与患者相同的 label_map 重映射，
    # 否则 GMM 原始分量下标与患者重排后的 subtype_label 不在同一坐标系。
    if label_map is not None:
        ctrl_labels = np.vectorize(label_map.get)(ctrl_labels)
    
    # 对照在各亚型中的分布
    ctrl_dist = pd.Series(ctrl_labels).value_counts().sort_index()
    pat_dist = shape_df[shape_df["group"] == "patient"]["subtype_label"].value_counts().sort_index()
    
    # 比较分布
    logger.info(f"  Subtype distribution comparison:")
    for k in range(n_subtypes):
        pat_pct = pat_dist.get(k, 0) / pat_dist.sum() * 100 if pat_dist.sum() > 0 else 0
        ctrl_pct = ctrl_dist.get(k, 0) / ctrl_dist.sum() * 100 if ctrl_dist.sum() > 0 else 0
        logger.info(f"    Subtype {k}: Patient={pat_pct:.1f}%, Control={ctrl_pct:.1f}%")
    
    # 卡方检验
    if len(ctrl_dist) == n_subtypes and len(pat_dist) == n_subtypes:
        contingency = pd.DataFrame({
            "patient": pat_dist.reindex(range(n_subtypes), fill_value=0),
            "control": ctrl_dist.reindex(range(n_subtypes), fill_value=0),
        })
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        logger.info(f"  Chi-square test: χ²={chi2:.2f}, p={p:.4f}")
        if p < 0.05:
            logger.info("  → Patient and control subtype distributions differ significantly")
        else:
            logger.info("  → No significant difference (subtypes may reflect general behavior patterns)")
    
    # [FIX v1.2] 增加效应量：最高严重度亚型的 Odds Ratio
    # [FIX v1.5] 动态确定高严重度亚型：按各亚型患者 step_height 中位数取最大者。
    # 此前硬编码 high_severity_subtype = 0，假设 Subtype 0 是高严重度组；
    # 但 label_map 按 step_height 降序排列后 Subtype 0 的 step 不一定最高
    # （本次运行 Subtype 0 实际为 recovery 最高），导致 OR 误标。
    pat_for_severity = shape_df[
        (shape_df["group"] == "patient") & shape_df["subtype_label"].notna()
    ]
    step_by_subtype = pat_for_severity.groupby("subtype_label")[
        "p_factor_severity_z_step_height"
    ].median()
    if len(step_by_subtype) > 0:
        high_severity_subtype = int(step_by_subtype.idxmax())
    else:
        high_severity_subtype = 0
    pat_in_high = (pat_dist.get(high_severity_subtype, 0) > 0)
    ctrl_in_high = (ctrl_dist.get(high_severity_subtype, 0) > 0)
    
    # 构建 2x2 表: [患者且高严重, 患者非高严重; 对照且高严重, 对照非高严重]
    a = pat_dist.get(high_severity_subtype, 0)
    b = pat_dist.sum() - a
    c = ctrl_dist.get(high_severity_subtype, 0)
    d = ctrl_dist.sum() - c
    
    if a > 0 and b > 0 and c > 0 and d > 0:
        # 计算 Odds Ratio 和 95% CI
        or_val = (a * d) / (b * c)
        log_or = np.log(or_val)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        logger.info(f"  High-severity subtype (Subtype {high_severity_subtype}) Odds Ratio (Patient vs Control): "
                    f"OR={or_val:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")

    # [FIX] 判别力检验：计算对照概率分布的熵
    ctrl_probs = gmm.predict_proba(X_ctrl)
    # 计算每个对照样本的熵
    entropy_per_ctrl = -np.sum(ctrl_probs * np.log2(ctrl_probs + 1e-10), axis=1)
    mean_entropy = np.mean(entropy_per_ctrl)
    max_entropy = np.log2(n_subtypes)
    
    # [FIX P2] 将对照组的亚型预测标签写回 shape_df，供下游分析使用
    ctrl_label_map = dict(zip(controls_valid["user_id_hash"], ctrl_labels))
    shape_df.loc[shape_df["group"] == "control", "subtype_label"] = \
        shape_df.loc[shape_df["group"] == "control", "user_id_hash"].map(ctrl_label_map)
    
    # [FIX v1.8] 将对照组的亚型概率列写回 shape_df（按 label_map 重排，与患者一致）。
    # 此前只写回硬标签 ctrl_labels，未写 ctrl_probs，导致下游 s10 亚型软匹配中
    # 对照的 subtype_{k}_prob 全为 NaN → fillna(0) → 全部 < 0.15 阈值被过滤，
    # Subtype-stratified DID 对照清空（14260981 → 0）。
    inv_label_map = {v: k for k, v in label_map.items()} if label_map else {i: i for i in range(n_subtypes)}
    for i in range(n_subtypes):
        ctrl_prob_map = dict(zip(controls_valid["user_id_hash"], ctrl_probs[:, inv_label_map[i]]))
        shape_df.loc[shape_df["group"] == "control", f"subtype_{i}_prob"] = \
            shape_df.loc[shape_df["group"] == "control", "user_id_hash"].map(ctrl_prob_map)
    
    logger.info(f"  Control subtype assignment entropy: {mean_entropy:.2f} / {max_entropy:.2f} bits")
    if mean_entropy > 0.85 * max_entropy:
        logger.warning("  WARNING: Control probabilities are highly uniform! "
                       "Subtypes may lack clinical specificity.")
# ======================== SUBTYPE PREDICTORS ========================

def analyze_subtype_predictors(shape_df: pd.DataFrame) -> pd.DataFrame:
    """
    分析哪些 Module 1 特征可以预测轨迹亚型。
    
    使用多分类逻辑回归 + 交叉验证。
    """
    logger.info("=" * 60)
    logger.info("Subtype Predictor Analysis")
    logger.info("=" * 60)
    
    # 仅有亚型标签的患者
    labeled = shape_df[
        (shape_df["group"] == "patient") & 
        (shape_df["subtype_label"].notna())
    ].copy()
    
    if len(labeled) < 100:
        logger.warning("  Too few labeled patients for predictor analysis")
        return pd.DataFrame()
    
    # 加载 Module 1 特征
    from config_module2 import FEATURES_DIR
    feat_path = os.path.join(FEATURES_DIR, "user_features_standardized.parquet")
    if not os.path.exists(feat_path):
        logger.warning("  Module 1 features not available")
        return pd.DataFrame()
    
    m1_features = pd.read_parquet(feat_path)
    
    # 合并
    # [FIX-5] 记录合并前样本量，诊断缩减原因
    n_before = len(labeled)
    merged = labeled.merge(m1_features, on="user_id_hash", how="inner")
    n_after = len(merged)
    merge_rate = n_after / max(1, n_before)

    logger.info(f"  Merged with Module 1 features: {n_after}/{n_before} "
            f"({merge_rate:.1%}) patients retained")

    # 诊断缩减原因
    if merge_rate < 0.8:
        lost_uids = set(labeled["user_id_hash"]) - set(merged["user_id_hash"])
        lost_in_m1 = lost_uids - set(m1_features["user_id_hash"])
        lost_by_nan = lost_uids & set(m1_features["user_id_hash"])
        
        logger.warning(f"  [Merge] Low merge rate ({merge_rate:.1%}):")
        logger.warning(f"    {len(lost_in_m1)} users not in Module 1 features")
        logger.warning(f"    {len(lost_by_nan)} users in Module 1 but lost in inner join (likely NaN in feature cols)")

    if len(merged) < 50:
        # [FIX P0] 降级策略：使用 left join + NaN 填充
        logger.warning(f"  [Merge] Inner join yielded {n_after} < 50 patients. "
                    f"Trying left join with zero-imputation...")
        merged = labeled.merge(m1_features, on="user_id_hash", how="left")
        # [FIX] 移除对未定义的 feature_cols 的引用，直接对所有数值列填0
        for col in merged.select_dtypes(include=[np.number]).columns:
            if col not in ["subtype_label"]: # 避免覆盖标签
                merged[col] = merged[col].fillna(0)
        logger.info(f"  [Merge] Left join: {len(merged)} patients")
        
        if len(merged) < 50:
            logger.error(f"  [Merge] Insufficient data even with left join ({len(merged)})")
            return pd.DataFrame()
    
    # 特征列 (排除元数据)
    from config_module2 import FEATURE_METADATA_FILE
    meta_path = os.path.join(FEATURES_DIR, "feature_selection.json")
    feature_cols = []
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            selection = json.load(f)
        feature_cols = selection.get("valid_features", [])
        if isinstance(feature_cols, list):
            feature_cols = [f for f in feature_cols if isinstance(f, str) and f in merged.columns]
    
    if not feature_cols:
        # [FIX-5] 尝试自动推断特征列
        logger.warning("  [Predictors] feature_selection.json not found or empty. "
                    "Auto-detecting numeric feature columns...")
        
        exclude_cols = {"user_id_hash", "group", "disorder_group", "t0_date", 
                        "is_retrospective", "subtype_label"}
        artifact_features = {
            "mental_health_sub_ratio", "support_sub_ratio",
            "subreddit_entropy", "subreddit_herfindahl",
            "primary_subreddit_ratio",
        }
        
        feature_cols = [
            c for c in m1_features.columns 
            if c not in exclude_cols 
            and c not in artifact_features
            and m1_features[c].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
        
        if not feature_cols:
            logger.error("  [Predictors] No valid feature columns found after auto-detection")
            return pd.DataFrame()
        
        logger.info(f"  [Predictors] Auto-detected {len(feature_cols)} features")
    
    # 排除选择伪影
    artifact_features = {
        "mental_health_sub_ratio", "support_sub_ratio",
        "subreddit_entropy", "subreddit_herfindahl",
        "primary_subreddit_ratio",
    }
    feature_cols = [c for c in feature_cols if c not in artifact_features]
    
    # 准备数据
    X = merged[feature_cols].values
    y = merged["subtype_label"].values.astype(int)
    
    # 去除 NaN
    valid = ~np.any(np.isnan(X), axis=1)
    X, y = X[valid], y[valid]
    
    if len(set(y)) < 2:
        logger.warning("  Only one subtype in data")
        return pd.DataFrame()
    
    # 多分类逻辑回归
    try:
        min_class_size = pd.Series(y).value_counts().min()
        cv_folds = min(5, min_class_size)

        if cv_folds < 2:
            logger.error(f"  Smallest subtype has only {min_class_size} samples, insufficient for CV")
            return pd.DataFrame()
        # [FIX v1.3] 修复类别不平衡：使用 balanced_accuracy 替代 accuracy，增加 CV 折数
        # [FIX v1.5] 移除 multi_class 参数：scikit-learn >=1.4 已从 LogisticRegressionCV
        # 移除该参数（lbfgs solver 自动处理多分类），传参导致 Predictor Analysis 崩溃。
        clf = LogisticRegressionCV(
            Cs=10, cv=cv_folds, penalty="l2", solver="lbfgs",
            max_iter=2000, 
            scoring="balanced_accuracy", # 对不平衡数据更鲁棒
            class_weight="balanced",     # 自动调整类别权重
            random_state=RANDOM_SEED, n_jobs=8,
        )
        clf.fit(X, y)
        
        # [FIX v1.2] 使用 Permutation Importance 替代朴素的系数绝对值均值
        # 这对特征共线性鲁棒，是顶刊标准做法
        from sklearn.inspection import permutation_importance
        
        perm_imp = permutation_importance(
            clf, X, y, n_repeats=20, 
            random_state=RANDOM_SEED, 
            scoring="balanced_accuracy", # 与模型评估指标保持一致
            n_jobs=8
        )
        
        predictor_df = pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": perm_imp.importances_mean,
            "importance_std": perm_imp.importances_std,
        }).sort_values("importance_mean", ascending=False)
        
        # 顶级预测因子
        logger.info(f"  Top subtype predictors (Permutation Importance):")
        for _, row in predictor_df.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        predictor_df.to_csv(
            os.path.join(STATS_DIR, "subtype_predictors.csv"), index=False)
        
        return predictor_df
        
    except Exception as e:
        logger.error(f"  Predictor analysis failed: {e}")
        return pd.DataFrame()


# ======================== DISORDER-SPECIFIC SUBTYPE ANALYSIS ========================

def disorder_specific_subtype_analysis(shape_df: pd.DataFrame) -> Dict:
    """
    分析每个疾病内部的亚型分布。
    
    不同疾病的亚型比例是否不同？
    """
    logger.info("=" * 60)
    logger.info("Disorder-Specific Subtype Analysis")
    logger.info("=" * 60)
    
    patients = shape_df[
        (shape_df["group"] == "patient") & 
        (shape_df["subtype_label"].notna())
    ]
    
    results = {}
    
    for disorder in DISORDER_GROUPS:
        dis_data = patients[patients["disorder_group"] == disorder]
        
        if len(dis_data) < 30:
            continue
        
        subtype_dist = dis_data["subtype_label"].value_counts().sort_index()
        total = len(dis_data)
        
        results[disorder] = {
            "n_patients": total,
            "subtype_distribution": subtype_dist.to_dict(),
            "subtype_percentages": {
                k: f"{v/total*100:.1f}%" 
                for k, v in subtype_dist.items()
            },
        }
        
        logger.info(f"  {disorder} (n={total}): {results[disorder]['subtype_percentages']}")
    
    # 疾病间亚型分布差异检验
    if len(results) >= 2:
        n_subtypes = int(patients["subtype_label"].max()) + 1
        contingency_data = []
        disorder_order = []
        
        for disorder in DISORDER_GROUPS:
            if disorder in results:
                row = [results[disorder]["subtype_distribution"].get(k, 0) 
                       for k in range(n_subtypes)]
                contingency_data.append(row)
                disorder_order.append(disorder)
        
        if len(contingency_data) >= 2:
            contingency = np.array(contingency_data)
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            logger.info(f"\n  Cross-disorder chi-square: χ²={chi2:.2f}, "
                       f"df={dof}, p={p:.4f}")
            if p < 0.05:
                logger.info("  → Subtype distributions differ significantly across disorders")
            else:
                logger.info("  → Failed to reject homogeneity of subtype distributions across disorders "
                           "(consistent with transdiagnostic patterns, but not a formal proof of equivalence)")
    
    with open(os.path.join(STATS_DIR, "disorder_subtype_analysis.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


# ======================== MAIN ========================

def run_trajectory_heterogeneity():
    """主管线: 轨迹异质性分析。"""
    logger.info("=" * 80)
    logger.info("MODULE 2 - Step 09: Trajectory Heterogeneity Analysis")
    logger.info("=" * 80)
    
    # 加载轨迹数据
    traj_path = os.path.join(TRAJECTORY_DIR, "user_trajectories.parquet")
    traj_df = pd.read_parquet(traj_path)
    logger.info(f"Loaded trajectories: {traj_df.shape}")

    # [FIX-A4] 防御性标准化 group 列
    traj_df["group"] = traj_df["group"].apply(
        lambda g: "patient" if g == "patient" else "control"
    )
    
    # 1. 提取轨迹形状特征
    shape_df = extract_trajectory_shape_features(traj_df)
    # 2. 发现轨迹亚型
    shape_df, subtype_interp = discover_trajectory_subtypes(shape_df)
    
    if not subtype_interp:
        logger.error("Subtype discovery failed!")
        return None
    
    # 3. 亚型预测因子
    predictor_df = analyze_subtype_predictors(shape_df)
    
    # 4. 疾病特异性亚型分析
    disorder_analysis = disorder_specific_subtype_analysis(shape_df)
    
    # 5. 保存
    shape_df.to_csv(
        os.path.join(STATS_DIR, "trajectory_subtypes.csv"), index=False)

    # [NEW] 聚合各亚型的轨迹数据，供直接绘制亚型对比图
    logger.info("Aggregating subtype trajectories for visualization...")
    # 合并轨迹和亚型标签
    traj_with_subtype = traj_df.merge(
        shape_df[["user_id_hash", "subtype_label"]], 
        on="user_id_hash", how="inner"
    )
    # 仅保留有标签且可靠的窗口
    traj_with_subtype = traj_with_subtype[
        traj_with_subtype["subtype_label"].notna() & traj_with_subtype["is_reliable"]
    ]
    
    if len(traj_with_subtype) > 0:
        subtype_agg = traj_with_subtype.groupby(
            ["subtype_label", "window_index", "days_from_t0_center"]
        )["p_factor_severity_z"].agg(["mean", "std", "count"]).reset_index()
        
        subtype_agg.columns = [
            "subtype_label", "window_index", "days_from_t0_center", 
            "p_factor_severity_z_mean", "p_factor_severity_z_std", "n"
        ]
        # 计算标准误
        subtype_agg["p_factor_severity_z_se"] = subtype_agg["p_factor_severity_z_std"] / np.sqrt(subtype_agg["n"])
        
        subtype_agg_path = os.path.join(STATS_DIR, "subtype_aggregated_trajectories.csv")
        subtype_agg.to_csv(subtype_agg_path, index=False)
        logger.info(f"Saved aggregated subtype trajectories: {subtype_agg_path}")

    with open(os.path.join(STATS_DIR, "subtype_profiles.json"), 'w') as f:
        json.dump(subtype_interp, f, indent=2, default=str)
    
    # === 综合报告 ===
    report = {
        "pipeline_version": "v1.0",
        "n_subtypes": len(subtype_interp),
        "subtype_names": {k: v["name"] for k, v in subtype_interp.items()},
        "subtype_sizes": {k: v["n_patients"] for k, v in subtype_interp.items()},
        "disorder_specific_results": {
            k: v["subtype_percentages"] for k, v in disorder_analysis.items()
        },
    }
    
    with open(os.path.join(STATS_DIR, "heterogeneity_report.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("=" * 80)
    logger.info("TRAJECTORY HETEROGENEITY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return shape_df, subtype_interp


if __name__ == "__main__":
    run_trajectory_heterogeneity()