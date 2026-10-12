"""NMH Figure 3 鈥?Temporal signature and the tested early-warning boundary.

Core conclusion: a reproducible, diagnosis-associated digital transition is
detectable around the self-reported diagnosis date and generalises across eras,
yet per-individual change-points are too sparse to support reliable early warning.

Panels
  a  Event-study coefficient trajectory around t0 (95% CI)
  b  GMM subtype trajectories (exploratory)
  c  Pre-t0 change-points by disorder (early-warning boundary)
  d  Per-disorder DID estimate (forest)
  e  Event-time profile replicated across three calendar eras
  f  Classifier temporal transfer (delta-AUROC vs atlas)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        DISORDERS, abbr, DIV_CMAP, SEQ_CAT, ACCENT_BLUE, ACCENT_RED)

B = "/path/to/project"
M2 = os.path.join(B, "module2_results", "statistics")
EV = os.path.join(M2, "event_study_results.csv")
ST = os.path.join(M2, "subtype_aggregated_trajectories.csv")
LEAD = os.path.join(M2, "signal_lead_time.csv")
DID = os.path.join(M2, "cross_disorder_did.csv")
TRAJ = os.path.join(B, "module4_results", "validation_r4", "r4_traj_temporal_split.csv")
CLF = os.path.join(B, "module4_results", "validation_r4", "r4_temporal_split_93train_results.csv")
AGG = os.path.join(B, "module2_results", "trajectories", "trajectory_aggregated_by_window.csv")

BAND_ORDER = ["far_pre", "mid_pre", "near_pre", "t0_surge",
              "near_post", "mid_post", "far_post"]
BAND_LABEL = ["Far-", "Mid-", "Near-", "t0", "Near+", "Mid+", "Far+"]
ERA_ORDER = ["E1_2010-2015", "E2_2016-2019", "E3_2020-2022"]
ERA_SHORT = {"E1_2010-2015": "E1 2010-15", "E2_2016-2019": "E2 2016-19",
             "E3_2020-2022": "E3 2020-22"}


def panel_a(ax):
    d = pd.read_csv(EV)
    d = d[d["variable"] == "p_factor_severity_z"].sort_values("days_from_t0")
    x = d["days_from_t0"].values
    y = d["coef"].values
    se = d["se"].values
    sig = d["p_fdr"].values < 0.05
    pre, post = x < 0, x >= 0
    PRE_C, POST_C = ACCENT_BLUE, ACCENT_RED
    for m, c, lab in [(pre, PRE_C, "Pre-t0"), (post, POST_C, "Post-t0")]:
        ax.errorbar(x[m], y[m], yerr=1.96 * se[m], fmt="o", ms=3.2, lw=0.9, capsize=1.6,
                    color=c, ecolor=c, elinewidth=0.7, label=lab)
    ax.scatter(x[sig], y[sig], s=20, facecolors="none", edgecolors="black", lw=0.5,
               zorder=5)
    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    ax.axhline(0, color="#CCCCCC", lw=0.6)
    i = int(np.nanargmax(y))
    ax.annotate(f"peak {y[i]:+.3f}", (x[i], y[i]), xytext=(4, 5),
                textcoords="offset points", fontsize=6.0, color="#333333")
    ax.text(0.02, 0.88, "Pre-t0", transform=ax.transAxes, color=PRE_C, fontsize=6.2,
            fontweight="bold", va="top", ha="left")
    ax.text(0.98, 0.88, "Post-t0", transform=ax.transAxes, color=POST_C, fontsize=6.2,
            fontweight="bold", va="top", ha="right")
    # NOTE: parallel-trends statistic (FAIL 17/22) is reported in the caption.
    ax.set_xlabel("Days from self-reported diagnosis (t0)")
    ax.set_ylabel("Case - comparison difference (z)")
    despine(ax)
    ax.set_title("Diagnosis-associated transition", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax_top, ax_bot):
    """Panel b: two stacked sub-panels sharing the x-axis (days from t0)."""
    CASE_C, COMP_C = "#6A4C93", "#2A9D8F"     # purple / teal (distinct from pre/post)
    # --- top: subtype trajectories ---
    d = pd.read_csv(ST)
    cols = {0.0: SEQ_CAT[2], 1.0: SEQ_CAT[4]}
    los, his = [], []
    for s in sorted(d["subtype_label"].unique()):
        g = d[d["subtype_label"] == s].sort_values("days_from_t0_center")
        x = g["days_from_t0_center"].values
        y = g["p_factor_severity_z_mean"].values
        se = g["p_factor_severity_z_se"].values
        ax_top.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=cols[s], alpha=0.18, lw=0)
        ax_top.plot(x, y, color=cols[s], lw=1.2)
        los.append(float((y - 1.96 * se).min())); his.append(float((y + 1.96 * se).max()))
    lo, hi = min(los), max(his); span = hi - lo if hi > lo else 1.0
    ax_top.set_ylim(min(-0.20, lo - 0.02), hi + 0.45 * span)
    ax_top.axvline(0, color="#999999", lw=0.6, ls="--")
    ax_top.axhline(0, color="#CCCCCC", lw=0.5)
    ax_top.set_ylabel("p-factor (z)", fontsize=7)
    ax_top.tick_params(labelbottom=False, labelsize=6)
    ax_top.text(0.02, 0.95, "Subtype 0", transform=ax_top.transAxes, color=cols[0.0],
                fontsize=6.0, va="top", ha="left")
    ax_top.text(0.02, 0.80, "Subtype 1", transform=ax_top.transAxes, color=cols[1.0],
                fontsize=6.0, va="top", ha="left")
    despine(ax_top)
    ax_top.set_title("Subtype trajectories", loc="left", fontsize=7.8)

    # --- bottom: case vs comparison (shared x) ---
    agg = pd.read_csv(AGG)
    los2, his2 = [], []
    for grp, c in [("patient", CASE_C), ("control", COMP_C)]:
        g = agg[agg["group"] == grp].sort_values("days_from_t0_center")
        y = g["p_factor_severity_z_mean"].values
        se = g["p_factor_severity_z_se"].values
        ax_bot.fill_between(g["days_from_t0_center"], y - 1.96 * se, y + 1.96 * se,
                            color=c, alpha=0.15, lw=0)
        ax_bot.plot(g["days_from_t0_center"], y, color=c, lw=1.0)
        los2.append(float((y - 1.96 * se).min())); his2.append(float((y + 1.96 * se).max()))
    lo2, hi2 = min(los2), max(his2); span2 = hi2 - lo2 if hi2 > lo2 else 1.0
    ax_bot.set_ylim(lo2 - 0.10 * span2, hi2 + 0.50 * span2)
    ax_bot.axvline(0, color="#999999", lw=0.6, ls="--")
    ax_bot.axhline(0, color="#CCCCCC", lw=0.5)
    ax_bot.set_xlabel("Days from t0")
    ax_bot.set_ylabel("p-factor (z)", fontsize=7)
    ax_bot.text(0.02, 0.95, "Case", transform=ax_bot.transAxes, color=CASE_C,
                fontsize=6.0, va="top", ha="left")
    ax_bot.text(0.02, 0.80, "Comparison", transform=ax_bot.transAxes, color=COMP_C,
                fontsize=6.0, va="top", ha="left")
    despine(ax_bot)
    ax_bot.set_title("Case vs comparison", loc="left", fontsize=7.8)
    panel_label(ax_top, "b")

    x_lo = float(min(d["days_from_t0_center"].min(), agg["days_from_t0_center"].min()))
    x_hi = float(max(d["days_from_t0_center"].max(), agg["days_from_t0_center"].max()))
    xs = x_hi - x_lo if x_hi > x_lo else 1.0
    ax_bot.set_xlim(x_lo - 0.02 * xs, x_hi + 0.02 * xs)


def panel_c(ax):
    d = pd.read_csv(LEAD)
    tot = int(d[d["disorder"] == "all"].iloc[0]["n_patients_with_pre_cp"])
    d = d[d["disorder"] != "all"].sort_values("median_lead_days")
    ypos = np.arange(len(d))
    x = d["median_lead_days"].values
    n = d["n_patients_with_pre_cp"].values
    for y, xi in zip(ypos, x):
        ax.plot([0, xi], [y, y], color="#C9D3E0", lw=0.9, zorder=1)
    ax.scatter(x, ypos, s=22 + 9 * n, color="#6E86A6", edgecolors="white", lw=0.7, zorder=3)
    for y, xi, ni in zip(ypos, x, n):
        ax.annotate(f"n={int(ni)}", (xi, y), xytext=(9, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=5.8, color="#444444")
    ax.set_yticks(ypos)
    ax.set_yticklabels([abbr(v) for v in d["disorder"]], fontsize=6.4)
    ax.set_xlabel("Median pre-t0 change-point lead time (days)")
    ax.set_xlim(0, max(x) * 1.35)
    despine(ax)
    ax.set_title(f"Early-warning boundary ({tot}/128k cases)", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(DID)
    order = [x for x in DISORDERS if x in set(d["disorder"])]
    d = d.set_index("disorder").loc[order]
    ypos = np.arange(len(order))[::-1]
    est = d["did_estimate"].values
    se = d["se"].values
    ax.errorbar(est, ypos, xerr=1.96 * se, fmt="o", ms=4.5, lw=1.0, capsize=2.2,
                color=ACCENT_BLUE, ecolor="#A9C0DC", mec="white", mew=0.5)
    ax.axvline(0, color="#999999", lw=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([abbr(x) for x in order], fontsize=6.4)
    lo_x = float(np.min(est - 1.96 * se)); hi_x = float(np.max(est + 1.96 * se))
    rng = hi_x - lo_x if hi_x > lo_x else 0.01
    ax.set_xlim(lo_x - 0.15 * rng, hi_x + 0.55 * rng)
    xr = ax.get_xlim()[1]
    for y, v in zip(ypos, est):                     # value column (right-aligned)
        ax.text(xr, y, f"{v:+.3f}", ha="right", va="center", fontsize=5.2, color="#444444")
    ax.set_xlabel("DID estimate (case x post)")
    despine(ax)
    ax.set_title("Per-disorder DID", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def panel_e(ax):
    d = pd.read_csv(TRAJ)
    band_pos = {b: i for i, b in enumerate(BAND_ORDER)}
    colors = [SEQ_CAT[1], SEQ_CAT[3], SEQ_CAT[4]]
    for era, c in zip(ERA_ORDER, colors):
        g = d[(d["era"] == era) & (d["band"].isin(BAND_ORDER))].copy()
        g["bp"] = g["band"].map(band_pos)
        g = g.sort_values("bp")
        ax.plot(g["bp"], g["p_factor_severity_z_diff"], "o-", ms=3.6, lw=1.2,
                color=c, label=ERA_SHORT[era])
    # NOTE: t0-surge values per era (0.105/0.107/0.109) are the replication result
    # and are given in the caption.
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(BAND_LABEL, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=5.8)
    ax.axhline(0, color="#CCCCCC", lw=0.6)
    ax.set_ylabel("Case - comparison diff (z)")
    lo = float(np.nanmin(d["p_factor_severity_z_diff"]))
    hi = float(np.nanmax(d["p_factor_severity_z_diff"]))
    span = hi - lo if hi > lo else 1.0
    ax.set_ylim(lo - 0.12 * span, hi + 0.55 * span)   # headroom for the legend
    ax.legend(fontsize=6.0, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#CCCCCC", framealpha=1.0, borderpad=0.6, labelspacing=0.45,
              handletextpad=1.0, handlelength=1.8)
    despine(ax)
    ax.set_title("Replicated across eras", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax):
    d = pd.read_csv(CLF)
    d["pair"] = d["train_era"] + " to " + d["test_era"]
    piv = d.pivot_table(index="disorder", columns="pair", values="delta_vs_atlas")
    cols = [p for p in ["E1 to E2", "E1 to E3", "E2 to E3"] if p in piv.columns]
    order = [x for x in DISORDERS if x in piv.index]
    M = piv.loc[order, cols].values.astype(float)
    v = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap=DIV_CMAP, vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=5.4,
                    color="white" if abs(M[i, j]) > 0.6 * v else "#333333")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([abbr(x) for x in order], fontsize=6.4)
    cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("delta-AUROC vs atlas", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.4)
    ax.set_title("Temporal transfer (93-feature atlas)", loc="left", fontsize=7.8)
    panel_label(ax, "f")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(165)))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                          wspace=0.42, hspace=0.42,
                          left=0.075, right=0.945, top=0.95, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]))
    sub = gs[0, 1].subgridspec(2, 1, hspace=0.45, height_ratios=[1, 1])
    b_top = fig.add_subplot(sub[0])
    b_bot = fig.add_subplot(sub[1], sharex=b_top)
    panel_b(b_top, b_bot)
    panel_c(fig.add_subplot(gs[0, 2]))
    panel_d(fig.add_subplot(gs[1, 0]))
    panel_e(fig.add_subplot(gs[1, 1]))
    panel_f(fig.add_subplot(gs[1, 2]))

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = os.path.join(MAIN_DIR, "fig3_temporal_boundary")
    require_matplotlib_panel_alignment(
        fig, json_out=out + ".alignment.json", overlay_svg=out + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".svg", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Exported:", out + ".{png,pdf,svg,tiff}")


if __name__ == "__main__":
    main()
