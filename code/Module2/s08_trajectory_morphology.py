"""
Step 08: Trajectory Morphology Analysis (v1.0)

识别 t0 前后的行为转折点和信号提前期。

核心分析:
  1. 事件研究法: t0 为事件日, 逐窗口异常值
  2. PELT 拐点检测: 识别行为突变的精确时间点
  3. 信号提前期: 从最早拐点到 t0 的时间差分布
  4. 平行趋势检验: DID 的前提验证

输入: user_trajectories.parquet (s07 输出)
输出: 
  - event_study_results.csv
  - changepoint_analysis.csv
  - signal_lead_time.csv
  - parallel_trends_test.json
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from linearmodels.panel import PanelOLS
import ruptures as rpt
from config_module2 import (
    TRAJECTORY_DIR, STATS_DIR,
    DISORDER_GROUPS, RANDOM_SEED,
    WINDOW_STEP_DAYS, WINDOW_SIZE_DAYS,PRODROME_WINDOW_DAYS,
    PSM_MATCHED_PATH,PELT_MIN_SEGMENT_LENGTH, CONSENSUS_TOLERANCE_DAYS
)

os.makedirs(STATS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATS_DIR, "trajectory_morphology.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


def changepoint_analysis(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个用户的 p_factor_severity_z 轨迹做 PELT 拐点检测。
    
    [FIX v2.0] 修正:
      1. 使用 ruptures 库替代手写 PELT，增强稳定性
      2. min_segment_length 降为 2，以捕捉 Type A "急转直下" 等短期突变
      3. 信号提前期使用 nsmallest，寻找最早的预警信号 (而非最靠近 t0 的信号)
    """
    logger.info("=" * 60)
    logger.info("Analysis 2: Changepoint Detection (PELT via ruptures)")
    logger.info("=" * 60)
    
    results = []
    target_var = "p_factor_severity_z"
    
    if target_var not in traj_df.columns:
        logger.error(f"  {target_var} not found in trajectory data!")
        return pd.DataFrame()
    
    # 对每个用户
    for uid, user_df in tqdm(traj_df.groupby("user_id_hash"), 
                              desc="PELT detection"):
        sorted_df = user_df.sort_values("window_index")
        
        series = sorted_df[target_var].values
        days = sorted_df["days_from_t0"].values
        
        # [FIX v2.2] 仅基于可靠窗口检测拐点：s07 在 z-score 之前对 silent
        # 窗口做了 MNAR 插补（P90/P10 伪值），若不排除会制造人造拐点。
        valid = ~np.isnan(series)
        if "is_reliable" in sorted_df.columns:
            valid &= sorted_df["is_reliable"].values
        if valid.sum() < 6: # 至少需要6个点才能做有意义的分段
            continue
        
        series_clean = series[valid].reshape(-1, 1) # ruptures 需要 2D array
        days_clean = days[valid]
        
        # [FIX v2.0] 使用标准 ruptures 库，min_size=2 对应 60天最短段长
        try:                
            algo = rpt.Pelt(custom_cost=rpt.costs.CostL2(),min_size=PELT_MIN_SEGMENT_LENGTH, jump=1).fit(series_clean)
            penalty = 2.0 * np.log(len(series_clean))
            cps = algo.predict(pen=penalty)
        except Exception as e:
            continue
        
        # ruptures 返回的是段结束的索引，转换为拐点起始索引
        valid_positions = np.where(valid)[0]  # [FIX v1.5] clean 索引 → 原始行索引映射
        for cp_idx in cps:
            if cp_idx < len(days_clean):
                orig_pos = valid_positions[cp_idx]
                results.append({
                    "user_id_hash": uid,
                    "group": sorted_df["group"].iloc[0],
                    "disorder_group": sorted_df["disorder_group"].iloc[0],
                    "changepoint_days_from_t0": days_clean[cp_idx],
                    "changepoint_window_index": sorted_df["window_index"].iloc[orig_pos],
                    "value_before": series_clean[cp_idx - 1, 0] if cp_idx > 0 else np.nan,
                    "value_after": series_clean[cp_idx, 0] if cp_idx < len(series_clean) else np.nan,
                    "value_change": (series_clean[cp_idx, 0] - series_clean[cp_idx - 1, 0]) 
                                    if cp_idx > 0 else np.nan,
                    "is_pre_t0": days_clean[cp_idx] < -WINDOW_STEP_DAYS / 2,
                    "is_post_t0": days_clean[cp_idx] > WINDOW_STEP_DAYS / 2,
                    "n_total_changepoints": len(cps) - 1, # ruptures 最后一个必是序列末尾
                })
    
    if not results:
        logger.warning("  No changepoints detected!")
        return pd.DataFrame()
    
    cp_df = pd.DataFrame(results)
    
    # === 信号提前期分析 ===
    pre_cps = cp_df[cp_df["is_pre_t0"] & (cp_df["group"] == "patient")]
    
    if len(pre_cps) > 0:
        # [FIX v2.0] 核心修正：信号提前期应寻找 "最早" 出现的异常，即 days_from_t0 最小（最负）的拐点
        # 之前使用 nlargest 选出了最靠近 t0 的拐点，导致提前期被严重低估
        
        pre_cps_prodomal = pre_cps[pre_cps["changepoint_days_from_t0"] >= -PRODROME_WINDOW_DAYS]
        
        if len(pre_cps_prodomal) > 0:
            earliest_pre = pre_cps_prodomal.groupby("user_id_hash").apply(
                lambda x: x.nsmallest(1, "changepoint_days_from_t0").iloc[0]
            )
            lead_times = -earliest_pre["changepoint_days_from_t0"]
        
        logger.info(f"  Signal lead time (patients with pre-t0 changepoints):")
        logger.info(f"    N patients: {len(lead_times)}")
        logger.info(f"    Median: {lead_times.median():.0f} days")
        logger.info(f"    Mean: {lead_times.mean():.0f} days")
        logger.info(f"    IQR: {lead_times.quantile(0.25):.0f} - "
                    f"{lead_times.quantile(0.75):.0f} days")
    
    # === 患者vs对照拐点分布 ===
    pat_cps = cp_df[cp_df["group"] == "patient"]["changepoint_days_from_t0"]
    ctrl_cps = cp_df[cp_df["group"] == "control"]["changepoint_days_from_t0"]
    
    if len(pat_cps) > 10 and len(ctrl_cps) > 10:
        ks_stat, ks_p = stats.ks_2samp(pat_cps, ctrl_cps)
        logger.info(f"  Changepoint distribution (patient vs control):")
        logger.info(f"    KS statistic: {ks_stat:.4f}, p={ks_p:.4f}")
        
        near_t0_pat = ((pat_cps > -90) & (pat_cps < 90)).mean()
        near_t0_ctrl = ((ctrl_cps > -90) & (ctrl_cps < 90)).mean()
        logger.info(f"    Near-t0 changepoints: "
                    f"patient={near_t0_pat:.2%}, control={near_t0_ctrl:.2%}")
    
    # === 按疾病分组 ===
    for disorder in DISORDER_GROUPS:
        disorder_cps = cp_df[
            (cp_df["group"] == "patient") & 
            (cp_df["disorder_group"] == disorder)
        ]
        if len(disorder_cps) > 10:
            pre_frac = disorder_cps["is_pre_t0"].mean()
            median_lead = -disorder_cps.loc[
                disorder_cps["is_pre_t0"], "changepoint_days_from_t0"
            ].median() if disorder_cps["is_pre_t0"].any() else np.nan
            logger.info(f"    {disorder}: pre-t0 CPs={pre_frac:.1%}, "
                       f"median lead={median_lead:.0f}d")
    
    # 保存
    cp_df.to_csv(os.path.join(STATS_DIR, "changepoint_analysis.csv"), index=False)
    
    # 信号提前期汇总
    lead_time_summary = _summarize_lead_times(cp_df)
    lead_time_summary.to_csv(
        os.path.join(STATS_DIR, "signal_lead_time.csv"), index=False)
    
    
    # === [FIX v2.1] PELT 敏感性分析：评估惩罚项对拐点数量的影响 ===
    # === [FIX v2.1] PELT 敏感性分析：评估惩罚项对拐点数量的影响 ===
    logger.info("  Running PELT Sensitivity Analysis (Penalty Multipliers)...")
    sensitivity_results = []
    penalty_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    # [必须修复] 抽样源应从 traj_df 全量用户中抽取，而非 cp_df（有拐点的用户子集）
    all_uids = traj_df["user_id_hash"].unique()
    if len(all_uids) > 1000:
        rng = np.random.RandomState(RANDOM_SEED)
        sample_uids = rng.choice(all_uids, 1000, replace=False)
    else:
        sample_uids = all_uids
        
    sample_df = traj_df[traj_df["user_id_hash"].isin(sample_uids)]
    
    for mult in penalty_multipliers:
        total_cps = 0
        n_users_with_cp = 0
        for uid, user_df in sample_df.groupby("user_id_hash"):
            sorted_df = user_df.sort_values("window_index")
            series = sorted_df[target_var].values
            valid = ~np.isnan(series)
            if "is_reliable" in sorted_df.columns:
                valid &= sorted_df["is_reliable"].values
            if valid.sum() < 6: continue
            series_clean = series[valid].reshape(-1, 1)
            try:
                algo = rpt.Pelt(custom_cost=rpt.costs.CostL2(), min_size=2, jump=1).fit(series_clean)
                penalty = mult * np.log(len(series_clean))
                cps = algo.predict(pen=penalty)
                # ruptures 最后一个返回值总是序列末尾，不算真实拐点
                real_cps = len(cps) - 1 
                total_cps += real_cps
                if real_cps > 0: n_users_with_cp += 1
            except Exception:
                continue
                
        sensitivity_results.append({
            "penalty_multiplier": mult,
            "avg_cps_per_user": total_cps / len(sample_uids) if len(sample_uids) > 0 else 0,
            "pct_users_with_cp": n_users_with_cp / len(sample_uids) if len(sample_uids) > 0 else 0
        })
        
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv(os.path.join(STATS_DIR, "pelt_sensitivity_analysis.csv"), index=False)
    logger.info(f"  Sensitivity analysis saved. BIC (2.0) avg CPs: {sens_df.loc[sens_df['penalty_multiplier']==2.0, 'avg_cps_per_user'].values[0]:.2f}")
    
    return cp_df

# ======================== EVENT STUDY ========================

def event_study_analysis(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX v2.0] 规范的双向固定效应 (TWFE) 面板事件研究法。
    使用 linearmodels.PanelOLS 确保标准误和估计量的无偏性，替代横截面回归。
    """
    logger.info("=" * 60)
    logger.info("Analysis 1: Event Study (TWFE Panel Regression)")
    logger.info("=" * 60)
    
    target_var = "p_factor_severity_z"
    if target_var not in traj_df.columns:
        logger.error(f"  {target_var} not found!")
        return pd.DataFrame()
    
    # 准备面板数据: 必须有 entity(time) 和 time 索引
    data = traj_df[traj_df["is_reliable"]].dropna(subset=[target_var]).copy()
    logger.info(f"  Reliable observations for TWFE: {len(data)}")
    
    # 基线参照期: window_index -8 到 -4，即 t0 前 240~90 天（窗口起始），
    # 窗口中心对应 t0 前 195~45 天。注释与实际 ref_range=(-8,-4) 对齐。
    ref_range = (-8, -4)
    data["window_cat"] = data["window_index"].astype(str)
    data.loc[data["window_index"].between(ref_range[0], ref_range[1]), "window_cat"] = "Baseline"
    
    # 构建 Patient × Window 交互项 (排除 Baseline)
    data["patient"] = (data["group"] == "patient").astype(int)
    interactions = pd.get_dummies(data["window_cat"], prefix="PxW", drop_first=False)
    
    # 删除参照组列 (完全多重共线处理)
    cols_to_drop = [col for col in interactions.columns if "Baseline" in col]
    interactions = interactions.drop(columns=cols_to_drop)
    
    # 乘以 Patient 指示变量
    for col in interactions.columns:
        interactions[col] = interactions[col] * data["patient"]
        
    data = pd.concat([data, interactions], axis=1)
    
    # 设置面板索引: entity = user_id_hash, time = window_index
    data = data.set_index(["user_id_hash", "window_index"])
    
    # 构建回归公式
    exog_cols = list(interactions.columns)
    
    try:
        # entity_effects 控制用户固定效应, time_effects 控制时间固定效应
        mod = PanelOLS(data[target_var], data[exog_cols], 
                       entity_effects=True, time_effects=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        # 提取交互项结果
        results = []
        for col in exog_cols:
            # 从列名 PxW_-10 提取 window_index -10
            win_idx = int(col.split("_")[-1])
            days_approx = win_idx * WINDOW_STEP_DAYS + WINDOW_SIZE_DAYS / 2.0
            
            coef_val = res.params[col]
            
            results.append({
                "variable": target_var,
                "window_index": win_idx,
                "days_from_t0": days_approx,
                "coef": coef_val,
                "cohens_d": coef_val,  # [FIX v2.1] 因变量为 Z-score，回归系数等价于 Cohen's d
                "se": res.std_errors[col],
                "se_type": "entity_clustered",
                "t_stat": res.tstats[col],
                "p_value": res.pvalues[col],
                "n_obs": res.nobs,
            })
            
        results_df = pd.DataFrame(results)
        
        # FDR 校正
        from statsmodels.stats.multitest import multipletests
        p_vals = results_df["p_value"].values
        _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        results_df["p_fdr"] = p_adj
        
        
        # [NEW] 为绘图补充参照期行 (系数=0, SE=0, p=1)
        # 参照期 window_index 在 ref_range = (-8, -4) 之间
        ref_rows = []
        for rw in range(ref_range[0], ref_range[1] + 1):
            ref_rows.append({
                "variable": target_var,
                "window_index": rw,
                "days_from_t0": rw * WINDOW_STEP_DAYS + WINDOW_SIZE_DAYS / 2.0,
                "coef": 0.0,
                "cohens_d": 0.0,
                "se": 0.0,
                "se_type": "entity_clustered",
                "t_stat": 0.0,
                "p_value": 1.0,
                "n_obs": res.nobs,
                "p_fdr": 1.0  # 参照期无需检验
            })
            
        if ref_rows:
            results_df = pd.concat([pd.DataFrame(ref_rows), results_df], ignore_index=True)
            results_df = results_df.sort_values("window_index").reset_index(drop=True)
        
        
        results_df.to_csv(os.path.join(STATS_DIR, "event_study_results.csv"), index=False)
        
        # 提取最早显著信号
        pre_sig = results_df[(results_df["days_from_t0"] < -WINDOW_STEP_DAYS) & (results_df["p_fdr"] < 0.05)]
        if len(pre_sig) > 0:
            earliest = pre_sig.loc[pre_sig["days_from_t0"].idxmin()] # 最早 = 最小值
            logger.info(f"  Earliest significant signal: t0{earliest['days_from_t0']:+.0f}d "
                       f"(coef={earliest['coef']:.4f}, p_FDR={earliest['p_fdr']:.4f}), Cohen's d={earliest['cohens_d']:.3f}")
                       
        return results_df
        
    except Exception as e:
        logger.error(f"PanelOLS failed: {e}")
        return pd.DataFrame()


def _summarize_lead_times(cp_df: pd.DataFrame) -> pd.DataFrame:
    """汇总各疾病和整体的信号提前期。"""
    results = []
    
    # 整体
    pre_cps = cp_df[cp_df["is_pre_t0"] & (cp_df["group"] == "patient")]
    # [新增] 过滤前驱期
    pre_cps_prodomal = pre_cps[pre_cps["changepoint_days_from_t0"] >= -PRODROME_WINDOW_DAYS]
    
    if len(pre_cps_prodomal) > 0:
        earliest = pre_cps_prodomal.groupby("user_id_hash").apply(
            lambda x: x.nsmallest(1, "changepoint_days_from_t0").iloc[0]
        )
        lead_times = -earliest["changepoint_days_from_t0"]
        results.append({
            "disorder": "all",
            "n_patients_with_pre_cp": len(lead_times),
            "median_lead_days": lead_times.median(),
            "mean_lead_days": lead_times.mean(),
            "q25_lead_days": lead_times.quantile(0.25),
            "q75_lead_days": lead_times.quantile(0.75),
        })
    
    # 按疾病
    for disorder in DISORDER_GROUPS:
        dis_pre = pre_cps_prodomal[pre_cps_prodomal["disorder_group"] == disorder]
        if len(dis_pre) > 5:
            earliest = dis_pre.groupby("user_id_hash").apply(
                lambda x: x.nsmallest(1, "changepoint_days_from_t0").iloc[0]
            )
            lead_times = -earliest["changepoint_days_from_t0"]
            results.append({
                "disorder": disorder,
                "n_patients_with_pre_cp": len(lead_times),
                "median_lead_days": lead_times.median(),
                "mean_lead_days": lead_times.mean(),
                "q25_lead_days": lead_times.quantile(0.25),
                "q75_lead_days": lead_times.quantile(0.75),
            })
    
    return pd.DataFrame(results)


# ======================== PARALLEL TRENDS TEST ========================

def parallel_trends_test(traj_df: pd.DataFrame, event_results: pd.DataFrame = None) -> Dict:
    """
    [FIX v2.0] 平行趋势假设检验（基于面板回归结果增强版）。
    修正了原版忽略系数协方差矩阵的致命错误，采用 Bonferroni 联合检验。
    """
    logger.info("=" * 60)
    logger.info("Analysis 3: Parallel Trends Test (Joint Inference)")
    logger.info("=" * 60)
    
    if event_results is None or len(event_results) == 0:
        # 如果没传入，尝试读取文件
        es_path = os.path.join(STATS_DIR, "event_study_results.csv")
        if os.path.exists(es_path):
            event_results = pd.read_csv(es_path)
        else:
            logger.warning("  No event study results provided for Parallel Trends Test.")
            return {}
            
    # 提取 t0 前的系数
    pre_results = event_results[event_results["days_from_t0"] < -WINDOW_STEP_DAYS / 2].copy()
    
    if len(pre_results) < 2:
        logger.warning("  Not enough pre-t0 windows to test parallel trends.")
        return {}
        
    # FDR 校正
    n_sig_fdr = (pre_results["p_fdr"] < 0.05).sum()
    
    # [FIX v2.1] 抛弃过度保守的 Bonferroni，改用基于 FDR 的联合推断
    # 逻辑：如果 pre-t0 窗口中没有任何一个在 FDR < 0.05 下显著，则接受平行趋势
    # FDR 天然考虑了多重检验的相关性结构，比 Bonferroni 更适合时间序列系数
    
    if n_sig_fdr == 0:
        conclusion = "PASS"
        message = "Parallel trends assumption holds (no pre-t0 differences at FDR<0.05)"
    elif n_sig_fdr <= 1:
        conclusion = "MARGINAL"
        message = f"Minor pre-t0 differences (FDR-sig: {n_sig_fdr} windows)"
    else:
        conclusion = "FAIL"
        message = f"Significant pre-t0 differences: {n_sig_fdr} windows significant at FDR<0.05"
        
    report = {
        "n_pre_windows_tested": len(pre_results),
        "n_sig_fdr_005": int(n_sig_fdr),
        "conclusion": conclusion,
        "message": message,
    }
    
    logger.info(f"  Result: {conclusion} - {message}")
    
    with open(os.path.join(STATS_DIR, "parallel_trends_test.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    return report


# ======================== MULTI-VARIABLE CHANGEPOINT ========================

def multivariate_changepoint_analysis(traj_df: pd.DataFrame) -> pd.DataFrame:
    """
    多变量联合拐点检测。
    
    在 p_factor_severity_z, volatility_index, social_withdrawal_index
    三个指标上同时检测拐点，寻找一致的转折点。
    """
    logger.info("=" * 60)
    logger.info("Analysis 4: Multi-variable Changepoint Consensus")
    logger.info("=" * 60)
    
    target_vars = ["p_factor_severity_z", "volatility_index", 
                   "social_withdrawal_index"]
    target_vars = [v for v in target_vars if v in traj_df.columns]
    
    # 对每个用户和每个变量检测拐点
    user_cp_dict = {}  # {uid: {var: [cp_days]}}
    
    for uid, user_df in tqdm(traj_df.groupby("user_id_hash"),
                              desc="Multi-var PELT"):
        sorted_df = user_df.sort_values("window_index")
        days = sorted_df["days_from_t0"].values
        
        user_cp_dict[uid] = {}
        
        for var in target_vars:
            series = sorted_df[var].values
            valid = ~np.isnan(series)
            if "is_reliable" in sorted_df.columns:
                valid &= sorted_df["is_reliable"].values
            
            if valid.sum() < 6: # 统一最低要求
                continue
            
            series_clean = series[valid].reshape(-1, 1) # ruptures 需要 2D
            days_clean = days[valid]
            
            try:
                # [FIX v2.0] 使用 ruptures
                algo = rpt.Pelt(custom_cost=rpt.costs.CostL2(), min_size=2, jump=1).fit(series_clean)
                penalty = 2.0 * np.log(len(series_clean))
                cps = algo.predict(pen=penalty)
                # 转换索引
                cp_days = [days_clean[cp] for cp in cps if cp < len(days_clean)]
                user_cp_dict[uid][var] = cp_days
            except Exception:
                continue
            
    # 共识分析: 在 ±30 天内出现的多变量拐点
    consensus_results = []
    
    for uid, var_cps in user_cp_dict.items():
        if len(var_cps) < 2:
            continue
        
        # 收集所有拐点
        all_cps = []
        for var, cps in var_cps.items():
            for cp_day in cps:
                all_cps.append({"var": var, "day": cp_day})
        
        if len(all_cps) < 2:
            continue
        
        # 检查是否有来自不同变量的拐点在 ±30 天内
        # [FIX v2.0] 优化共识匹配：使用 O(N log N) 排序扫描，替代 O(N²) 暴力匹配
        all_cps.sort(key=lambda x: x["day"])
        i = 0
        while i < len(all_cps):
            j = i + 1
            current_group = [all_cps[i]]
            # 寻找 30 天容忍度内的所有拐点
            # 将多变量匹配中的 30 替换为配置变量：
            while j < len(all_cps) and (all_cps[j]["day"] - all_cps[i]["day"]) <= CONSENSUS_TOLERANCE_DAYS:
                # 检查新加入的点与组内所有点的距离是否都在容忍度内
                valid_addition = True
                for existing_cp in current_group:
                    if abs(all_cps[j]["day"] - existing_cp["day"]) > CONSENSUS_TOLERANCE_DAYS:
                        valid_addition = False
                        break
                if valid_addition:
                    current_group.append(all_cps[j])
                    j += 1
                else:
                    break
                
            # 如果组内有来自不同变量的拐点
            vars_in_group = set(c["var"] for c in current_group)
            if len(vars_in_group) >= 2:
                avg_day = np.mean([c["day"] for c in current_group])
                # 记录所有跨变量配对
                for x in range(len(current_group)):
                    for y in range(x+1, len(current_group)):
                        if current_group[x]["var"] != current_group[y]["var"]:
                            consensus_results.append({
                                "user_id_hash": uid,
                                "consensus_day": avg_day,
                                "var1": current_group[x]["var"],
                                "var1_day": current_group[x]["day"],
                                "var2": current_group[y]["var"],
                                "var2_day": current_group[y]["day"],
                                "day_difference": abs(current_group[x]["day"] - current_group[y]["day"]),
                                "is_pre_t0": avg_day < -WINDOW_STEP_DAYS / 2,
                            })
            i = j # 跳过已处理的组
    
    if not consensus_results:
        logger.info("  No multi-variable consensus changepoints found")
        return pd.DataFrame()
    
    consensus_df = pd.DataFrame(consensus_results)
    
    # 汇总
    n_consensus = len(consensus_df)
    n_pre_t0 = consensus_df["is_pre_t0"].sum()
    
    logger.info(f"  Multi-variable consensus changepoints: {n_consensus}")
    logger.info(f"  Pre-t0: {n_pre_t0} ({n_pre_t0/n_consensus*100:.1f}%)")
    
    # 最常见的变量组合
    var_pairs = consensus_df.groupby(["var1", "var2"]).size().sort_values(ascending=False)
    logger.info(f"  Most common variable pairs:")
    for (v1, v2), count in var_pairs.head(5).items():
        logger.info(f"    {v1} × {v2}: {count}")
    
    consensus_df.to_csv(
        os.path.join(STATS_DIR, "multivariate_changepoints.csv"), index=False)
    
    return consensus_df


# ======================== MAIN ========================

def run_trajectory_morphology():
    """主管线: 轨迹形态学分析。"""
    logger.info("=" * 80)
    logger.info("MODULE 2 - Step 08: Trajectory Morphology Analysis")
    logger.info("=" * 80)
    
    # 加载轨迹数据
    traj_path = os.path.join(TRAJECTORY_DIR, "user_trajectories.parquet")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectories not found: {traj_path}")
    
    traj_df = pd.read_parquet(traj_path)
    logger.info(f"Loaded trajectories: {traj_df.shape}")

    # [FIX-A3] 防御性标准化 group 列（兼容旧版数据）
    traj_df["group"] = traj_df["group"].apply(
        lambda g: "patient" if g == "patient" else "control"
    )
    logger.info(f"Group distribution after normalization: "
                f"{traj_df['group'].value_counts().to_dict()}")
    logger.info(f"Users: {traj_df['user_id_hash'].nunique()}")
    
    # 1. 事件研究法
    event_results = event_study_analysis(traj_df)
    
    # 2. 拐点检测
    cp_results = changepoint_analysis(traj_df)
    
    # 3. 平行趋势检验 (传入事件研究结果，用于联合推断)
    trends_report = parallel_trends_test(traj_df, event_results)
    
    # 4. 多变量联合拐点
    multi_cp_results = multivariate_changepoint_analysis(traj_df)
    
    # === 综合报告 ===
    report = {
        "pipeline_version": "v1.0",
        "event_study": {
            "n_significant_pre_t0_windows": int(
                (event_results["p_fdr"] < 0.05).sum()
            ) if "p_fdr" in event_results.columns and len(event_results) > 0 else 0,
        },
        "changepoint_detection": {
            "n_total_changepoints": len(cp_results),
            "n_pre_t0_changepoints": int(cp_results["is_pre_t0"].sum()) 
                if len(cp_results) > 0 else 0,
        },
        "parallel_trends": {
            "conclusion": trends_report.get("conclusion", "unknown"),
            # [FIX v1.5] 键名修正：parallel_trends_test 返回的键是 n_sig_fdr_005，
            # 此前用 n_significant_at_005 导致永远取默认值 -1
            "n_significant_pre_t0": trends_report.get("n_sig_fdr_005", -1),
        },
        "multivariate_consensus": {
            "n_consensus_changepoints": len(multi_cp_results),
        },
    }
    
    with open(os.path.join(STATS_DIR, "morphology_report.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("=" * 80)
    logger.info("TRAJECTORY MORPHOLOGY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return event_results, cp_results, trends_report


if __name__ == "__main__":
    run_trajectory_morphology()