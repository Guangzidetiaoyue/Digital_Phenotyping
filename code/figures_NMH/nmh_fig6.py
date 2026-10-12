"""NMH Figure 6 - Composition-induced temporal bias (CITB) audit.

Message: apparent calendar-time trends in digital signals are driven by event-time
composition drift; band adjustment and window regression recover the true trend.

Panels:
  a  Line        : naive vs band-adjusted annual series (p-factor, volatility)
  b  Stacked area: event-time band composition by calendar year
  c  Scatter     : estimated vs true trend (3 estimators x 3 scenarios, 1:1 line)
  d  Heatmap     : estimator bias by scenario
  e  Forest      : band-adjusted Kendall tau per signal
  f  Line        : simulated calendar-time series (naive vs adjusted, A-C)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        DIV_CMAP, SEQ_CMAP, ACCENT_BLUE, ACCENT_RED, NEUTRAL, GREEN,
                        ORANGE, PURPLE, GOLD)

B = "/path/to/project"
D4 = os.path.join(B, "module4_results", "data")
S4 = os.path.join(B, "module4_results", "statistics")
R4 = os.path.join(B, "module4_results", "validation_r4")
POOLED = os.path.join(D4, "digital_signal_annual_pooled.csv")
ADJ = os.path.join(D4, "digital_signal_annual_band_adjusted.csv")
BYBAND = os.path.join(D4, "digital_signal_annual_by_band.csv")
SIM = os.path.join(R4, "citb_simulation_results.csv")
CORR = os.path.join(S4, "trend_corrected_results.json")

METRICS = [("p_factor_severity_z", "p-factor", "#355C8A"),
           ("volatility_index", "volatility", ORANGE)]
BAND_ORDER = ["far_pre", "mid_pre", "near_pre", "t0_surge", "near_post", "mid_post",
              "far_post"]
SCEN_FULL = {"A_true_trend": "A: trend", "B_no_trend_only_composition":
             "B: composition only", "C_trend_plus_composition": "C: trend + composition"}
ESTS = [("naive_slope", "naive", ACCENT_RED, "o", "none", -0.0016),
        ("band_adjusted_slope", "band", ACCENT_BLUE, "o", "filled", 0.0),
        ("window_reg_cal_coef", "window-reg", NEUTRAL, "^", "filled", 0.0016)]
XT = [2010, 2014, 2018, 2022]


def _naive_adj(base):
    p = pd.read_csv(POOLED)
    piv = p.pivot_table(index="year", columns="group", values=f"{base}_mean",
                        aggfunc="mean")
    naive = piv["patient"] - piv["control"]
    adj = pd.read_csv(ADJ).set_index("year")[f"{base}_diff"]
    return pd.DataFrame({"naive": naive, "adj": adj}).loc[2010:2022]


def panel_a(ax):
    for base, lab, c in METRICS:
        df = _naive_adj(base)
        ax.plot(df.index, df["naive"], color=c, lw=0.9, ls="--", alpha=0.8)
        ax.plot(df.index, df["adj"], color=c, lw=1.7)
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    handles = [Line2D([], [], color=METRICS[0][2], lw=0.9, ls="--",
                      label="p-factor (naive)"),
               Line2D([], [], color=METRICS[0][2], lw=1.7, label="p-factor (adjusted)"),
               Line2D([], [], color=METRICS[1][2], lw=0.9, ls="--",
                      label="volatility (naive)"),
               Line2D([], [], color=METRICS[1][2], lw=1.7, label="volatility (adjusted)")]
    ax.legend(handles=handles, fontsize=5.6, loc="lower right", frameon=True,
              framealpha=1.0, edgecolor="#CCCCCC", handlelength=1.6, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.25)
    ax.set_ylabel("Annual patient - comparison gap")
    ax.set_xlim(2010, 2022)
    ax.set_xticks(XT)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Naive (dashed) vs\nband-adjusted", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(BYBAND)
    d = d[d["band"].isin(BAND_ORDER)]
    piv = d.pivot_table(index="year", columns="band", values="n_patient",
                        aggfunc="sum").reindex(columns=BAND_ORDER).loc[2010:2022]
    share = piv.div(piv.sum(axis=1), axis=0)
    cols = [SEQ_CMAP(i / (len(BAND_ORDER) - 1)) for i in range(len(BAND_ORDER))]
    ax.stackplot(share.index, *[share[b].values for b in BAND_ORDER], colors=cols, lw=0)
    ax.set_xlim(2010, 2022)
    ax.set_xticks(XT)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of patients")
    handles = [Patch(facecolor=cols[i], label=BAND_ORDER[i].replace("_", " "))
               for i in range(len(BAND_ORDER))]
    ax.legend(handles=handles, fontsize=5.6, ncol=2, loc="lower left", frameon=True,
              framealpha=1.0, edgecolor="#CCCCCC", handlelength=0.9, handletextpad=0.3,
              labelspacing=0.22, borderpad=0.3)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Event-time composition\ndrift", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(SIM)
    gv_all = d["true_gamma"].values
    ax.plot([0, 0.033], [0, 0.033], color="#BBBBBB", lw=0.8, ls="--", zorder=2)
    for col, lab, c, mk, fill, dx in ESTS:
        ax.scatter(gv_all + dx, d[col].values, s=26, marker=mk, zorder=4,
                   facecolors=c if fill == "filled" else "none", edgecolors=c, lw=0.9,
                   label=lab)
    encl = [(0.03, "A: trend", PURPLE, (0.0245, 0.0195), "left", "top"),
            (0.000, "B: comp only", GOLD, (0.0065, -0.0012), "left", "center"),
            (0.01, "C: trend+comp", GREEN, (0.0005, 0.0195), "left", "bottom")]
    for gv, lab, ec, xy, ha, va in encl:
        m = np.isclose(gv_all, gv)
        pts = np.array([[gv + dx, d.loc[m, col].values[0]] for col, _, _, _, _, dx in ESTS])
        cx, cy = pts.mean(0)
        r = np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)) + 0.0013
        ax.add_patch(Circle((cx, cy), r, facecolor=ec, edgecolor=ec, lw=0.7,
                            alpha=0.16, zorder=1))
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ec, lw=0.7, zorder=3))
        ax.annotate(lab, xy=(cx, cy), xycoords="data", xytext=xy, textcoords="data",
                    ha=ha, va=va, fontsize=5.6, color="#333333", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#BBBBBB", alpha=1.0,
                                    shrinkA=1.0, shrinkB=6.0))
    ax.set_xlabel("True trend slope")
    ax.set_ylabel("Estimated slope")
    ax.set_xlim(-0.004, 0.036)
    ax.set_ylim(-0.004, 0.036)
    ax.legend(fontsize=5.6, loc="upper left", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Estimator accuracy\n(dashed = identity)", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(SIM)
    ests = [("naive_bias", "naive"), ("band_adjusted_bias", "band"),
            ("window_reg_cal_bias", "window-reg")]
    M = np.array([[r[e] for e, _ in ests] for _, r in d.iterrows()], float).T
    vmax = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=DIV_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.4f}", ha="center", va="center", fontsize=5.0,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "#333333")
    for i in range(1, M.shape[0]):
        ax.axhline(i - 0.5, color="white", lw=0.8)
    for j in range(1, M.shape[1]):
        ax.axvline(j - 0.5, color="white", lw=0.8)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(["A\ntrend", "B\ncomposition\nonly", "C\ntrend +\ncomposition"],
                       fontsize=5.0)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([lab for _, lab in ests], fontsize=5.6)
    ax.set_title("Estimator bias by scenario", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def panel_e(ax):
    g = json.load(open(CORR))["metrics"]
    short = {"p_factor_severity_z_diff": "p-factor", "posting_intensity_diff": "posting",
             "social_withdrawal_index_diff": "withdrawal",
             "volatility_index_diff": "volatility"}
    rows = [(short.get(k, k), g[k]["kendall_tau"], g[k]["kendall_p"]) for k in g.keys()]
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    for y, (lab, tau, p) in zip(ys, rows):
        c = ACCENT_BLUE if p < 0.05 else "#999999"
        ax.plot([0, tau], [y, y], color=c, lw=1.1, zorder=3)
        ax.scatter([tau], [y], s=18, color=c, zorder=4)
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("Band-adjusted Kendall tau")
    taus = [r[1] for r in rows]
    ax.set_xlim(min(taus) - 0.12, max(taus) + 1.20)
    for y, (lab, tau, p) in zip(ys, rows):
        ax.text(0.98, y, f"p={p:.3f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=5.0, color="#333333", zorder=5)
    ax.tick_params(axis="both", pad=5)
    despine(ax)
    ax.set_title("Corrected trends", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax):
    nb = pd.read_csv(os.path.join(R4, "citb_series_B_no_trend_only_composition_naive.csv"))
    bb = pd.read_csv(os.path.join(R4, "citb_series_B_no_trend_only_composition_band.csv"))
    xn = nb.iloc[:, 0].values.astype(float)
    yn = nb.iloc[:, -1].values.astype(float)
    xb = bb.iloc[:, 0].values.astype(float)
    yb = bb.iloc[:, -1].values.astype(float)
    mean = float(np.mean(np.concatenate([yn, yb])))
    ax.axhline(mean, color="#999999", lw=0.8, ls="--", zorder=2)
    ax.plot(xn, yn, color=ACCENT_RED, lw=1.0, marker="o", ms=2.0, zorder=4,
            label="naive")
    ax.plot(xb, yb, color=ACCENT_BLUE, lw=1.2, marker="s", ms=1.8, zorder=4,
            label="adjusted")
    ax.plot([], [], color="#999999", lw=0.8, ls="--", label="true level")
    ax.legend(fontsize=5.6, loc="lower right", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", handlelength=1.4, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.25)
    ax.set_xlabel("Simulated calendar year")
    ax.set_ylabel("Simulated signal")
    ax.set_xticks([int(v) for v in xn if int(v) % 4 == 0])
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Scenario B: composition only", loc="left", fontsize=7.8)
    panel_label(ax, "f")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(170)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.62, hspace=0.42,
                          left=0.11, right=0.955, top=0.95, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
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
    out = os.path.join(MAIN_DIR, "fig6_citb_audit")
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
