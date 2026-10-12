"""NMH Extended Data Figure ED10 - Assumptions and provenance (4 panels).
(a) MNAR dose-response; (b) CITB scenarios A/C; (c) provenance schema; (d) placebo DID null."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, NEUTRAL,
                        ACCENT_BLUE, ACCENT_RED)

R4 = "/path/to/project/module4_results/validation_r4"
M2 = "/path/to/project/module2_results/statistics"


def panel_a(ax):
    j = json.load(open(os.path.join(R4, "r4_mnar_sensitivity_results.json")))
    xs = np.array(list(j["dose"].values()) if isinstance(j["dose"], dict) else j["dose"],
                  float)
    ys = np.array(list(j["response"].values()) if isinstance(j["response"], dict)
                  else j["response"], float)
    n = min(len(xs), len(ys))
    ax.plot(xs[:n], ys[:n], "-o", color=ACCENT_BLUE, ms=3.0, lw=1.1)
    ax.set_xlabel("Silent-window dose")
    ax.set_ylabel("Response")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("MNAR dose-response", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    for scen, lab, off in [("A_true_trend", "A: trend", 0.0),
                           ("C_trend_plus_composition", "C: trend+comp", 0.6)]:
        nn = pd.read_csv(os.path.join(R4, f"citb_series_{scen}_naive.csv"))
        bb = pd.read_csv(os.path.join(R4, f"citb_series_{scen}_band.csv"))
        x = nn.iloc[:, 0].values.astype(float)
        y1 = nn.iloc[:, -1].values.astype(float)
        y2 = bb.iloc[:, -1].values.astype(float)
        y1 = y1 - y1.mean() + off
        y2 = y2 - y2.mean() + off
        ax.plot(x, y1, color=ACCENT_RED, lw=0.9)
        ax.plot(x, y2, color=ACCENT_BLUE, lw=1.0, ls=(0, (3, 2)))
        ax.text(x.min(), off, lab, fontsize=5.0, ha="left", va="bottom", color="#333333")
    ax.set_yticks([])
    ax.set_xlabel("Simulated year")
    ax.tick_params(axis="x", pad=5, labelsize=5.6)
    despine(ax, left=True)
    ax.set_title("CITB A/C (red naive, blue band)", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    ax.set_axis_off()
    boxes = [(0.01, 0.66, 0.31, 0.22, "Raw posts"),
             (0.345, 0.66, 0.31, 0.22, "PSM cohort"),
             (0.68, 0.66, 0.31, 0.22, "Features"),
             (0.01, 0.16, 0.31, 0.22, "M1 atlas"),
             (0.345, 0.16, 0.31, 0.22, "M2 temporal"),
             (0.68, 0.16, 0.31, 0.22, "M3/M4")]
    for x, y, w, h, t in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F2F5F9", edgecolor="#B4C4D6",
                               lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.0,
                color="#333333")
    for x0, x1 in [(0.32, 0.345), (0.655, 0.68)]:
        ax.annotate("", xy=(x1, 0.77), xytext=(x0, 0.77),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=NEUTRAL))
    for x0 in (0.165, 0.50, 0.835):
        ax.annotate("", xy=(x0, 0.40), xytext=(x0, 0.64),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=NEUTRAL))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Provenance schema", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M2, "placebo_did_distribution.csv"))
    v = pd.to_numeric(d["placebo_did"], errors="coerce").dropna().values
    ax.hist(v, bins=30, color=ACCENT_BLUE, edgecolor="white", lw=0.3)
    ax.axvline(0, color=ACCENT_RED, lw=0.8, ls="--")
    ax.set_xlabel("Placebo DID estimate")
    ax.set_ylabel("Count")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Placebo DID null", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.45, left=0.10, right=0.965,
                          top=0.94, bottom=0.09)
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
    out = os.path.join(ED_DIR, "ed10_assumptions_provenance")
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
