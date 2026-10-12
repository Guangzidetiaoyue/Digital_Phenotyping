"""NMH Extended Data Figure ED9 - Population detail (4 panels).
(a) long-term trend per disorder; (b) withdrawal COVID shift; (c) STL decomposition;
(d) breakpoint sensitivity."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, ACCENT_BLUE, ACCENT_RED, NEUTRAL)

M4 = "/path/to/project/module4_results/statistics"


def panel_a(ax):
    d = pd.read_csv(os.path.join(M4, "trend_by_disorder.csv")).set_index("disorder")
    ds = [x for x in DISORDERS if x in d.index]
    ds.sort(key=lambda x: d.loc[x, "tau"])
    ys = np.arange(len(ds))
    for y, x in zip(ys, ds):
        tau, p = d.loc[x, "tau"], d.loc[x, "p"]
        c = ACCENT_BLUE if p < 0.05 else NEUTRAL
        ax.plot([0, tau], [y, y], color=c, lw=1.1, zorder=3)
        ax.scatter([tau], [y], s=18, color=c, zorder=4)
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(x) for x in ds], fontsize=5.6)
    ax.set_xlabel("Mann-Kendall tau")
    ax.margins(x=0.2)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Long-term trend", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(os.path.join(M4, "covid_its_corrected_results.csv"))
    s = d[d["metric"] == "social_withdrawal_index_diff"].set_index("disorder")
    ds = [x for x in DISORDERS if x in s.index]
    ds.sort(key=lambda x: s.loc[x, "b_post_shift"])
    ys = np.arange(len(ds))
    for y, x in zip(ys, ds):
        b, se = s.loc[x, "b_post_shift"], s.loc[x, "se_post_shift"]
        sig = s.loc[x, "p_post_shift_fdr"] < 0.05
        ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y], color=NEUTRAL, lw=0.9, zorder=2)
        ax.scatter([b], [y], s=18, color=ACCENT_BLUE if sig else "white",
                   edgecolors=ACCENT_BLUE, lw=0.8, zorder=4)
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(x) for x in ds], fontsize=5.6)
    ax.set_xlabel("COVID level shift (withdrawal)")
    ax.margins(x=0.15)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Withdrawal COVID shift", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M4, "trend_decomposition.csv"))
    x = pd.to_datetime(d["period"])
    ax.plot(x, d["trend"], color=ACCENT_BLUE, lw=1.2, label="trend")
    ax.plot(x, d["seasonal"], color=NEUTRAL, lw=0.8, alpha=0.9, label="seasonal")
    ax.set_ylabel("Component")
    ax.legend(fontsize=5.0, loc="best", frameon=False)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("STL decomposition", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M4, "covid_its_sensitivity_breakpoint.csv"))
    g = d.groupby("breakpoint")["b_post_shift"].mean()
    bps = list(g.index)
    xs = np.arange(len(bps))
    ax.plot(xs, g.values, "-o", color=ACCENT_BLUE, ms=2.6, lw=1.0)
    ax.axhline(0, color="#999999", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in bps], rotation=45, fontsize=5.0)
    ax.set_ylabel("Level shift (mean)")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Breakpoint sensitivity", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.48, left=0.11, right=0.965,
                          top=0.94, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = os.path.join(ED_DIR, "ed9_population_detail")
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
