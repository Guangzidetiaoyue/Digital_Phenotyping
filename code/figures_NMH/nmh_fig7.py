"""NMH Figure 7 - External validation: behaviour versus language.

Message: of the digital signals, only behavioural volatility validates against external
epidemiological benchmarks (suicide rate); language-driven signals show raw correlations
that do not survive detrending and are composition-sensitive.

Provenance: every number drawn comes from the result files (no fitting, detrending or
correlation is recomputed here):
  module4_results/statistics/lead_lag_comparison.csv          (annual series, a)
  module4_results/statistics/lead_lag_corrected_results.csv   (lag r + CI, b)
  module4_results/validation_r4/external_robustness_results_corrected.csv (a,c,d,e,f)
  module4_results/validation_r4/external_robustness_results_naive.csv     (f)

Panels:
  a  Scatter + inset : volatility vs suicide (annual series) + raw/detrended r inset
  b  Dot + CI        : lead-lag r at lags -1/0/+1 per signal
  c  Dot matrix      : signal x lag landscape (area |r|, colour sign, fill p<0.05)
  d  Interval        : LOYO range (bar) + raw r (dot) per signal
  e  Dot pair        : first-difference and detrended r per signal
  f  Dumbbell        : raw r with bootstrap CI, naive vs corrected
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        ACCENT_BLUE, ACCENT_RED, NEUTRAL, GREEN, PURPLE, DARK_BLUE)

B = "/path/to/project"
S4 = os.path.join(B, "module4_results", "statistics")
R4 = os.path.join(B, "module4_results", "validation_r4")
COMP = os.path.join(S4, "lead_lag_comparison.csv")
LL = os.path.join(S4, "lead_lag_corrected_results.csv")
CORR = os.path.join(R4, "external_robustness_results_corrected.csv")
NAIVE = os.path.join(R4, "external_robustness_results_naive.csv")

SIGS = [("volatility_index_diff", "volatility", ACCENT_BLUE),
        ("posting_intensity_diff", "posting", ACCENT_RED),
        ("social_withdrawal_index_diff", "withdrawal", GREEN),
        ("p_factor_severity_z_diff", "p-factor", PURPLE)]
SIG_LAB = {s: lab for s, lab, _ in SIGS}
SIG_COL = {s: c for s, _, c in SIGS}
LAG_SET = [-3, -2, -1, 0, 1, 2, 3]


def _row(df, sig, ext="suicide_rate"):
    return df[(df["signal"] == sig) & (df["external"] == ext)].iloc[0]


def panel_a(ax):
    d = pd.read_csv(COMP).dropna(subset=["suicide", "volatility_index_diff"])
    x = d["suicide"].values.astype(float)
    y = d["volatility_index_diff"].values.astype(float)
    ax.scatter(x, y, s=22, color=ACCENT_BLUE, edgecolors="white", lw=0.6, zorder=4)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 40)
    ax.plot(xs, a + b * xs, color=DARK_BLUE, lw=1.0, ls="-", alpha=0.9)
    r = pd.read_csv(CORR)
    row = _row(r, "volatility_index_diff")
    ax.text(0.95, 0.09, f"raw: r = {row['raw_r']:.2f}, p = {row['raw_p']:.3f}\n"
            f"n = {int(row['n_years'])}, 95% CI [{row['bootstrap_ci_lo']:.2f}, "
            f"{row['bootstrap_ci_hi']:.2f}]",
            transform=ax.transAxes, fontsize=5.2, va="bottom", ha="right",
            color="#333333", linespacing=1.35)
    ax.set_xlabel("Suicide rate (per 100,000)")
    ax.set_ylabel("Volatility gap")
    ax.margins(0.14)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Volatility vs suicide (fit)", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(LL)
    d = d[d["external"] == "suicide_rate"]
    lags = [-1, 0, 1]
    sigs = ["volatility_index_diff", "posting_intensity_diff"]
    ys = {lg: i for i, lg in enumerate(lags)}
    for sig in sigs:
        s = d[d["digital_signal"] == sig].set_index("lag")
        c = SIG_COL[sig]
        off = -0.16 if sig == "volatility_index_diff" else 0.16
        for lg in lags:
            if lg not in s.index:
                continue
            r = s.loc[lg, "r"]
            lo = s.loc[lg, "ci_lo"]
            hi = s.loc[lg, "ci_hi"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [ys[lg] + off, ys[lg] + off], color=c, lw=0.8,
                        alpha=0.8, zorder=3)
            dy = -5 if off < 0 else 5
            ax.annotate(f"{r:.2f}", (r, ys[lg] + off), xytext=(0, dy),
                        textcoords="offset points", fontsize=5.0,
                        va="top" if dy < 0 else "bottom", ha="center", color="#333333",
                        zorder=5)
            ax.scatter([r], [ys[lg] + off], s=20, color=c, zorder=4,
                       label=SIG_LAB[sig] if lg == lags[0] else None)
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([f"lag {lg:+d}" for lg in lags], fontsize=5.6)
    ax.set_ylim(-0.6, len(lags) - 0.15)
    ax.set_xlim(-1.1, 1.1)
    ax.text(1.05, len(lags) - 0.30, "volatility", color=ACCENT_BLUE, fontsize=5.6,
            ha="right", va="center")
    ax.text(-1.05, len(lags) - 0.30, "posting", color=ACCENT_RED, fontsize=5.6,
            ha="left", va="center")
    ax.set_xlabel("r at each lag (95% CI)")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Lead-lag vs suicide", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(CORR)
    sigs = [s for s, _, _ in SIGS]
    for i, sig in enumerate(sigs):
        row = _row(d, sig)
        L = json.loads(row["lags"])
        for j, lg in enumerate(LAG_SET):
            if str(lg) not in L:
                continue
            r = L[str(lg)]["r"]
            p = L[str(lg)]["p"]
            sigp = p < 0.05
            col = ACCENT_BLUE if r > 0 else ACCENT_RED
            ax.scatter([j], [i], s=8 + 60 * abs(r), marker="o",
                       facecolors=col if sigp else "none", edgecolors=col, lw=0.8,
                       zorder=4)
            ax.text(j - 0.36, i + 0.37, f"{r:+.1f}", ha="left", va="top",
                    fontsize=5.0, color="#333333", zorder=5)
    ax.set_xticks(range(len(LAG_SET)))
    ax.set_xticklabels([str(l) for l in LAG_SET], fontsize=5.6)
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels([SIG_LAB[s] for s in sigs], fontsize=5.6)
    ax.set_xlabel("Lag (years)")
    ax.set_xlim(-0.6, len(LAG_SET) - 0.4)
    ax.set_ylim(-0.6, len(sigs) - 0.4)
    ax.tick_params(axis="both", pad=5)
    despine(ax)
    ax.set_title("Correlation landscape", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(CORR)
    rows = []
    for s, lab, c in SIGS:
        row = _row(d, s)
        rows.append((lab, row["loyo_min_r"], row["loyo_max_r"], row["raw_r"],
                     row["bootstrap_ci_lo"], row["bootstrap_ci_hi"]))
    rows.sort(key=lambda t: t[2] - t[1])
    ys = np.arange(len(rows))[::-1]
    for y, (lab, lo, hi, r, cl, ch) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="#BBBBBB", lw=3.2, solid_capstyle="round",
                zorder=2)
        ax.plot([cl, ch], [y, y], color=NEUTRAL, lw=0.8, zorder=3)
        ax.scatter([r], [y], s=20, color=ACCENT_BLUE, zorder=4)
        ax.text(max(hi, ch) + 0.10, y, f"{hi - lo:.2f}", va="center", ha="left",
                fontsize=5.6, color="#333333")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("r (stability)")
    ax.set_xlim(-1.75, 1.55)
    handles = [Line2D([], [], color="#BBBBBB", lw=3.0, label="LOYO range"),
               Line2D([], [], color=NEUTRAL, lw=1.0, label="95% CI"),
               Line2D([], [], marker="o", ls="none", color=ACCENT_BLUE, ms=3.4,
                      label="raw r")]
    ax.legend(handles=handles, fontsize=5.0, loc="upper left", frameon=True,
              framealpha=0.95, edgecolor="#CCCCCC", handlelength=1.6, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Stability", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def panel_e(ax):
    d = pd.read_csv(CORR)
    rows = []
    for s, lab, c in SIGS:
        row = _row(d, s)
        rows.append((lab, row["firstdiff_r"], row["detrended_r"]))
    rows.sort(key=lambda t: t[2])
    ys = np.arange(len(rows))
    for y, (lab, rf, rd) in zip(ys, rows):
        ax.plot([rf, rd], [y, y], color="#CCCCCC", lw=1.0, zorder=2)
        ax.scatter([rf], [y], s=20, facecolors="none", edgecolors=ACCENT_BLUE, lw=0.9,
                   zorder=4)
        ax.scatter([rd], [y], s=20, color=ACCENT_BLUE, zorder=4)
        ax.text(max(rf, rd) + 0.09, y, f"{rd:+.2f}", va="center", ha="left",
                fontsize=5.6, color="#333333")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("r (robustness)")
    ax.set_xlim(-1.55, 0.95)
    handles = [Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                      markeredgecolor=ACCENT_BLUE, ms=3.4, label="first-difference"),
               Line2D([], [], marker="o", ls="none", color=ACCENT_BLUE, ms=3.4,
                      label="detrended")]
    ax.legend(handles=handles, fontsize=5.0, loc="upper left", frameon=True,
              framealpha=0.95, edgecolor="#CCCCCC", handletextpad=0.4, borderpad=0.3,
              labelspacing=0.25)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Robustness transformations", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax):
    c = pd.read_csv(CORR)
    n = pd.read_csv(NAIVE)
    rows = []
    for s, lab, col in SIGS:
        rc = _row(c, s)
        rn = _row(n, s)
        rows.append((lab, rn["raw_r"], rc["raw_r"], rn["bootstrap_ci_lo"],
                     rn["bootstrap_ci_hi"], rc["bootstrap_ci_lo"], rc["bootstrap_ci_hi"],
                     col))
    rows.sort(key=lambda t: t[2])
    ys = np.arange(len(rows))[::-1]
    for y, (lab, rn, rc, nl, nh, cl, ch, col) in zip(ys, rows):
        ax.plot([nl, nh], [y + 0.14, y + 0.14], color=NEUTRAL, lw=0.7)
        ax.plot([cl, ch], [y - 0.14, y - 0.14], color=NEUTRAL, lw=0.7)
        ax.plot([rn, rc], [y + 0.14, y - 0.14], color="#CCCCCC", lw=0.8, zorder=2)
        ax.scatter([rn], [y + 0.14], s=20, facecolors="none", edgecolors=col, lw=0.9,
                   zorder=4)
        ax.scatter([rc], [y - 0.14], s=20, color=col, zorder=4)
        ax.annotate(f"{rn:.2f}", (rn, y + 0.14), xytext=(0, 4),
                    textcoords="offset points", fontsize=5.0, va="bottom", ha="center",
                    color="#333333")
        ax.annotate(f"{rc:.2f}", (rc, y - 0.14), xytext=(0, -4),
                    textcoords="offset points", fontsize=5.0, va="top", ha="center",
                    color="#333333")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("r (naive / corrected)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.75, len(rows) - 0.15)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Naive (top) vs corrected (bottom)", loc="left", fontsize=7.8)
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
    out = os.path.join(MAIN_DIR, "fig7_external_validation")
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
