"""
Step 11: Publication-Quality Visualization for Module 2

Generates Nature/Science/PNAS-level composite figures summarizing
longitudinal trajectory, morphology, heterogeneity, and causal findings.

Design Principles:
  1. Minimal chartjunk (no gridlines, minimal borders, clean axes)
  2. Color-blind friendly palettes (Okabe-Ito / Nature default)
  3. High information density via multi-panel layouts
  4. Vector output (PDF) + high-res raster (300 DPI PNG)
  5. Strict typography (Helvetica/Arial, 6-8pt)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from typing import Dict, Optional

from config_module2 import (
    TRAJECTORY_DIR, STATS_DIR, M2_FIGURES_DIR,
    DISORDER_DISPLAY_NAMES, HITOP_COLORS, WINDOW_STEP_DAYS, WINDOW_SIZE_DAYS
)

os.makedirs(M2_FIGURES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
#  GLOBAL STYLE CONFIGURATION (Nature/Science Standard)
# =====================================================================

def setup_nature_style():
    """配置符合顶刊要求的 matplotlib 全局样式"""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        
        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        
        # Lines & Marks
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Layout
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# Color Palettes
PAT_CTRL_PALETTE = {"patient": "#E64B35", "control": "#4DBBD5"}  # Nature Red & Cyan
SUBTYPE_PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]  # Okabe-Ito variant
ACCENT_COLOR = "#666666"

setup_nature_style()


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def add_panel_label(ax, label: str, x=-0.15, y=1.05):
    """添加图面版面标签"""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


def plot_event_study_coefs(ax, es_df: pd.DataFrame, color_dict: Dict):
    """绘制带误差棒的事件研究系数图"""
    if es_df.empty:
        return
        
    # 区分基线、pre、post
    baseline_mask = es_df["coef"] == 0.0
    pre_mask = es_df["days_from_t0"] < -WINDOW_STEP_DAYS
    post_mask = es_df["days_from_t0"] > WINDOW_STEP_DAYS
    
    # 画基线参考线
    ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
    ax.axvline(0, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
    
    # 填充显著区域背景
    sig_pre = es_df[pre_mask & (es_df["p_fdr"] < 0.05)]
    sig_post = es_df[post_mask & (es_df["p_fdr"] < 0.05)]
    
    if not sig_pre.empty:
        ax.fill_between(sig_pre["days_from_t0"], -10, 10, 
                        color='#E64B35', alpha=0.05, zorder=0)
    if not sig_post.empty:
        ax.fill_between(sig_post["days_from_t0"], -10, 10, 
                        color='#E64B35', alpha=0.05, zorder=0)
        
    # 画系数和误差棒
    for mask, color in [(pre_mask, '#4DBBD5'), (post_mask, '#E64B35')]:
        subset = es_df[mask]
        if subset.empty: continue
        
        ax.errorbar(subset["days_from_t0"], subset["coef"], 
                    yerr=1.96 * subset["se"], 
                    fmt='o', markersize=3, capsize=2, capthick=0.8,
                    color=color, ecolor=color, elinewidth=0.8, zorder=3)
                    
        # 显著点用实心圆突出
        sig_mask = subset["p_fdr"] < 0.05
        if sig_mask.any():
            ax.scatter(subset.loc[sig_mask, "days_from_t0"], 
                      subset.loc[sig_mask, "coef"],
                      s=25, color=color, zorder=4, edgecolors='black', linewidths=0.5)


# =====================================================================
#  FIGURE 1: MAIN TRAJECTORY & MORPHOLOGY
# =====================================================================

def generate_figure1():
    """
    Figure 1: Longitudinal Trajectories and Morphology
    (a) Main trajectories of p-factor severity (Patient vs Control)
    (b) Event study coefficients (Dynamic causal effects)
    (c) Signal lead time distribution
    """
    logger.info("Generating Figure 1: Main Trajectory & Morphology...")
    
    # Load Data
    agg_path = os.path.join(TRAJECTORY_DIR, "trajectory_aggregated_by_window.csv")
    es_path = os.path.join(STATS_DIR, "event_study_results.csv")
    cp_path = os.path.join(STATS_DIR, "changepoint_analysis.csv")
    
    if not os.path.exists(agg_path):
        logger.error("Missing aggregated trajectory data!"); return
        
    agg_df = pd.read_csv(agg_path)
    
    # Setup Figure (2x2 layout with merged bottom-right)
    fig = plt.figure(figsize=(7.2, 5.5))  # Nature full width ~180mm
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1],
                           hspace=0.4, wspace=0.35)
    
    # --- Panel A: Trajectory Plot ---
    ax1 = fig.add_subplot(gs[0, :])  # Span top row
    
    for grp, color in PAT_CTRL_PALETTE.items():
        grp_data = agg_df[(agg_df["group"] == grp) & (agg_df["window_index"] >= -24) & (agg_df["window_index"] <= 24)]
        if grp_data.empty: continue
        
        ax1.plot(grp_data["days_from_t0_center"], grp_data["p_factor_severity_z_mean"], 
                color=color, label=grp.capitalize(), linewidth=1.8)
        
        # Ribbon (SEM)
        ax1.fill_between(
            grp_data["days_from_t0_center"],
            grp_data["p_factor_severity_z_mean"] - grp_data["p_factor_severity_z_se"],
            grp_data["p_factor_severity_z_mean"] + grp_data["p_factor_severity_z_se"],
            color=color, alpha=0.15, linewidth=0
        )
        
    ax1.axvline(0, color='#333333', linewidth=1.2, linestyle='--', zorder=1)
    ax1.set_xlim(-730, 730)
    ax1.set_xlabel("Days from t₀")
    ax1.set_ylabel("P-Factor Severity (Z-score)")
    ax1.legend(frameon=False, loc='upper left')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    # Annotation for t0
    ax1.annotate('t₀ (First MH post)', xy=(0, ax1.get_ylim()[1]*0.9), 
                fontsize=6.5, ha='center', color='#333333')
    
    add_panel_label(ax1, 'a')
    
    # --- Panel B: Event Study ---
    ax2 = fig.add_subplot(gs[1, 0])
    if os.path.exists(es_path):
        es_df = pd.read_csv(es_path)
        plot_event_study_coefs(ax2, es_df, PAT_CTRL_PALETTE)
        ax2.set_xlim(-600, 600)
        ax2.set_ylim(-0.3, 0.8)
        ax2.set_xlabel("Days from t₀")
        ax2.set_ylabel("Effect size (Cohen's d)")
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax2.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax2.transAxes)
    
    add_panel_label(ax2, 'b')
    
    # --- Panel C: Signal Lead Time Distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    if os.path.exists(cp_path):
        cp_df = pd.read_csv(cp_path)
        pre_cps = cp_df[(cp_df["is_pre_t0"]) & (cp_df["group"] == "patient")]
        
        if not pre_cps.empty:
            # 绘制最早拐点的分布
            earliest_cp = pre_cps.groupby("user_id_hash")["changepoint_days_from_t0"].min()
            lead_days = -earliest_cp
            
            sns.kdeplot(lead_days, ax=ax3, fill=True, color='#E64B35', alpha=0.4, linewidth=1.5)
            ax3.axvline(lead_days.median(), color='#333333', linewidth=1.5, linestyle='--')
            
            # Annotation
            med_val = lead_days.median()
            ax3.annotate(f'Median: {med_val:.0f}d', 
                        xy=(med_val, ax3.get_ylim()[1]*0.8), 
                        xytext=(med_val + 60, ax3.get_ylim()[1]*0.8),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2),
                        fontsize=7)
                        
        ax3.set_xlabel("Signal Lead Time (Days)")
        ax3.set_ylabel("Density")
        ax3.set_xlim(0, 720)
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax3.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax3.transAxes)
        
    add_panel_label(ax3, 'c')
    
    # Save
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig1_trajectory_morphology.pdf"))
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig1_trajectory_morphology.png"), dpi=600)
    plt.close(fig)
    logger.info("Figure 1 saved.")


# =====================================================================
#  FIGURE 2: HETEROGENEITY & SUBTYPES
# =====================================================================

def generate_figure2():
    """
    Figure 2: Trajectory Heterogeneity
    (a) Subtype trajectories
    (b) Subtype feature radar chart
    (c) Subtype prevalence by disorder
    """
    logger.info("Generating Figure 2: Heterogeneity & Subtypes...")
    
    # Load Data
    sub_agg_path = os.path.join(STATS_DIR, "subtype_aggregated_trajectories.csv")
    sub_feat_path = os.path.join(STATS_DIR, "trajectory_subtypes.csv")
    sub_prof_path = os.path.join(STATS_DIR, "subtype_profiles.json")
    
    if not os.path.exists(sub_agg_path):
        logger.error("Missing subtype aggregated data!"); return
        
    sub_agg = pd.read_csv(sub_agg_path)
    
    with open(sub_prof_path, 'r') as f:
        profiles = json.load(f)
        
    n_subtypes = len(profiles)
    sub_colors = SUBTYPE_PALETTE[:n_subtypes]
    
    # Setup Figure
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.4, wspace=0.4)
    
    # --- Panel A: Subtype Trajectories ---
    ax1 = fig.add_subplot(gs[0, :])
    
    for i in range(n_subtypes):
        s_data = sub_agg[sub_agg["subtype_label"] == i]
        if s_data.empty: continue
        
        name = profiles[f"subtype_{i}"]["name"]
        ax1.plot(s_data["days_from_t0_center"], s_data["p_factor_severity_z_mean"], 
                color=sub_colors[i], label=f"S{i}: {name}", linewidth=1.8)
                
        ax1.fill_between(
            s_data["days_from_t0_center"],
            s_data["p_factor_severity_z_mean"] - s_data["p_factor_severity_z_se"],
            s_data["p_factor_severity_z_mean"] + s_data["p_factor_severity_z_se"],
            color=sub_colors[i], alpha=0.1, linewidth=0
        )
        
    ax1.axvline(0, color='#333333', linewidth=1.2, linestyle='--', zorder=1)
    ax1.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
    ax1.set_xlim(-730, 730)
    ax1.set_xlabel("Days from t₀")
    ax1.set_ylabel("P-Factor Severity (Z-score)")
    ax1.legend(frameon=False, ncol=2, loc='upper left')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a')
    
    # --- Panel B: Radar Chart of Subtype Features ---
    ax2 = fig.add_subplot(gs[1, 0], polar=True)
    
    # Extract core features for radar
    radar_feats = {
        "Step Height": "p_factor_severity_z_step_height",
        "Pre Slope": "p_factor_severity_z_slope_pre",
        "Post Slope": "p_factor_severity_z_slope_post",
        "Volatility": "p_factor_severity_z_volatility_pre",
        "Turning Angle": "p_factor_severity_z_turning_angle"
    }
    
    if os.path.exists(sub_feat_path):
        sub_df = pd.read_csv(sub_feat_path)
        sub_df_pat = sub_df[sub_df["group"] == "patient"]
        
        # Compute medians and normalize to [0, 1] for radar
        radar_data = {}
        max_vals = {}
        for fname, fcol in radar_feats.items():
            if fcol in sub_df_pat.columns:
                max_vals[fname] = sub_df_pat[fcol].abs().max()
                
        for i in range(n_subtypes):
            s_medians = []
            for fname, fcol in radar_feats.items():
                if fcol in sub_df_pat.columns:
                    med_val = sub_df_pat[sub_df_pat["subtype_label"] == i][fcol].median()
                    norm_val = med_val / max_vals[fname] if max_vals[fname] > 0 else 0
                    s_medians.append(norm_val)
                else:
                    s_medians.append(0)
            radar_data[i] = s_medians
            
        # Plot Radar
        angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
        angles += angles[:1]
        
        for i in range(n_subtypes):
            values = radar_data[i] + radar_data[i][:1]
            ax2.plot(angles, values, 'o-', linewidth=1.5, color=sub_colors[i], 
                    label=f"S{i}", markersize=3)
            ax2.fill(angles, values, alpha=0.1, color=sub_colors[i])
            
        ax2.set_thetagrids(np.degrees(angles[:-1]), list(radar_feats.keys()))
        ax2.set_ylim(-1, 1)
        ax2.set_rticks([-0.5, 0, 0.5, 1.0])
        ax2.set_yticklabels(["-0.5", "0", "0.5", "1.0"], fontsize=5, color='grey')
        ax2.spines['polar'].set_visible(False)
        ax2.grid(color='#cccccc', linewidth=0.5)
        
    add_panel_label(ax2, 'b', x=-0.2, y=1.15)
    
    # --- Panel C: Stacked Bar of Disorder Distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    dis_data = []
    for i in range(n_subtypes):
        dis_dist = profiles[f"subtype_{i}"]["disorder_distribution"]
        n_total = profiles[f"subtype_{i}"]["n_patients"]
        for dis, count in dis_dist.items():
            dis_data.append({
                "Subtype": f"S{i}",
                "Disorder": DISORDER_DISPLAY_NAMES.get(dis, dis),
                "Percentage": (count / n_total) * 100 if n_total > 0 else 0
            })
            
    if dis_data:
        dis_df = pd.DataFrame(dis_data)
        pivot_df = dis_df.pivot(index="Subtype", columns="Disorder", values="Percentage").fillna(0)
        
        # Plot
        pivot_df.plot(kind='barh', stacked=True, ax=ax3, 
                     color=list(HITOP_COLORS.values())[:pivot_df.shape[1]],
                     edgecolor='white', linewidth=0.5)
                     
        ax3.set_xlabel("Proportion (%)")
        ax3.set_ylabel("")
        ax3.set_yticklabels([profiles[f"subtype_{i}"]["name"] for i in range(n_subtypes)], fontsize=6)
        ax3.legend(frameon=False, ncol=2, loc='lower right', fontsize=5)
        
    add_panel_label(ax3, 'c')
    
    # Save
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig2_heterogeneity_subtypes.pdf"))
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig2_heterogeneity_subtypes.png"), dpi=600)
    plt.close(fig)
    logger.info("Figure 2 saved.")


# =====================================================================
#  FIGURE 3: CAUSAL ATTRIBUTION
# =====================================================================

def generate_figure3():
    """
    Figure 3: Causal Attribution
    (a) Placebo test distribution
    (b) Dose-response scatter with OLS fit
    """
    logger.info("Generating Figure 3: Causal Attribution...")
    
    # Load Data
    placebo_path = os.path.join(STATS_DIR, "placebo_did_distribution.csv")
    dose_path = os.path.join(STATS_DIR, "dose_response_scatter_data.csv")
    
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.4)
    
    # --- Panel A: Placebo Test ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    if os.path.exists(placebo_path):
        plac_df = pd.read_csv(placebo_path)
        true_did_path = os.path.join(STATS_DIR, "placebo_test_results.json")
        
        # Distribution
        sns.histplot(plac_df["placebo_did"], ax=ax1, kde=True, 
                    color='#4DBBD5', alpha=0.4, edgecolor=None, linewidth=0)
        
        if os.path.exists(true_did_path):
            with open(true_did_path, 'r') as f:
                true_res = json.load(f)
            true_did = true_res["true_did"]
            p_val = true_res["placebo_p_value"]
            
            ax1.axvline(true_did, color='#E64B35', linewidth=2, linestyle='-')
            ax1.annotate(f'True DID = {true_did:.2f}\np = {p_val:.3f}', 
                        xy=(true_did, ax1.get_ylim()[1]*0.8), 
                        xytext=(true_did + 0.1, ax1.get_ylim()[1]*0.8),
                        fontsize=6.5, color='#E64B35', fontweight='bold')
                        
    ax1.set_xlabel("Placebo DID Estimate")
    ax1.set_ylabel("Density")
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, 'a')
    
    # --- Panel B: Dose-Response ---
    ax2 = fig.add_subplot(gs[1, 1])
    
    if os.path.exists(dose_path):
        dose_df = pd.read_csv(dose_path)
        
        if "dose_auc" in dose_df.columns and "response_post_mean" in dose_df.columns:
            # Scatter
            ax2.scatter(dose_df["dose_auc"], dose_df["response_post_mean"], 
                       s=10, color='#4DBBD5', alpha=0.4, edgecolors='none')
            
            # OLS Fit Line (using partial residual if available, else raw)
            x_col = "dose_auc"
            y_col = "response_post_mean" 
            fit_col = "predicted_response"
            
            if fit_col in dose_df.columns:
                plot_df = dose_df.dropna(subset=[x_col, fit_col]).sort_values(x_col)
                ax2.plot(plot_df[x_col], plot_df[fit_col], 
                        color='#E64B35', linewidth=2, linestyle='-')
            else:
                # Fallback linear fit
                from scipy.stats import linregress
                valid = dose_df[[x_col, y_col]].dropna()
                slope, intercept, r, p, se = linregress(valid[x_col], valid[y_col])
                x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
                ax2.plot(x_range, slope * x_range + intercept, 
                        color='#E64B35', linewidth=2, linestyle='-')
                        
            ax2.set_xlabel("Pre-t₀ Cumulative Severity (AUC)")
            ax2.set_ylabel("Post-t₀ Severity Level")
            ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
            
    add_panel_label(ax2, 'b')
    
    # Save
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig3_causal_attribution.pdf"))
    fig.savefig(os.path.join(M2_FIGURES_DIR, "fig3_causal_attribution.png"), dpi=600)
    plt.close(fig)
    logger.info("Figure 3 saved.")


# =====================================================================
#  MAIN
# =====================================================================

def run_visualization():
    """Execute all figure generation pipelines."""
    logger.info("=" * 80)
    logger.info("MODULE 2 - Step 11: Publication-Quality Visualization")
    logger.info("=" * 80)
    
    generate_figure1()
    generate_figure2()
    generate_figure3()
    
    logger.info("All figures generated successfully.")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_visualization()