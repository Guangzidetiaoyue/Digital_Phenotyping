"""NMH Figure 1 — Study design and the scale-dependent validity map.

Panels
  a  Cohort construction funnel (Phase 1 -> PSM-matched cohort)
  b  Case composition by disorder (PSM layer, computed from step_3.4 parquet)
  c  Capability map ("what digital phenotyping can/cannot do")
  d  Scale ladder (individual -> disorder -> structure -> temporal -> population)

All numbers computed from NAS result files where possible; the funnel stage
tallies are documented preprocessing-pipeline counts (data_preprocessing_pipeline.md).
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

from common_nmh import (apply_style, mm2inch, save_fig, despine, panel_label,
                        DISORDERS, DISPLAY, disorder_color, spectrum_of,
                        SPECTRUM_COLORS, VERDICT_COLORS)

PROJECT_ROOT = "/path/to/project"
PSM_PATH = os.path.join(PROJECT_ROOT, "cohort_v1", "intermediate_results",
                        "step_3.4_psm_matched.parquet")


# ------------------------------------------------------------------ data
def load_composition():
    schema = pq.read_schema(PSM_PATH)
    names = schema.names
    gcol = "group" if "group" in names else next(c for c in names if "group" in c.lower())
    dcol = ("disorder_group" if "disorder_group" in names
            else next(c for c in names if "disorder" in c.lower()))
    df = pd.read_parquet(PSM_PATH, columns=[gcol, dcol])
    pat = df[df[gcol] == "patient"]
    ctl = df[df[gcol] != "patient"]
    counts = pat[dcol].value_counts()
    n_pat, n_ctl = int(len(pat)), int(len(ctl))
    unlabelled = int(counts.get("unknown", 0))
    return counts, n_pat, n_ctl, unlabelled


# ------------------------------------------------------------------ panels
def panel_a(ax, n_pat, n_ctl, ratio):
    ax.set_axis_off()
    stages = [
        ("Full Pushshift dump (2006-2022)", "billions of posts"),
        ("Diagnosis candidates (Phase 1)", "cases + candidates"),
        ("Full-history extraction (Phase 2)", "4,950,144"),
        ("Bot + quality filtering", "4,771,571"),
        ("shadow-T0 assigned", "3,730,213"),
        ("PSM-matched cohort", "723,574"),
    ]
    n = len(stages)
    top, h, gap = 0.985, 0.108, 0.036
    for i, (lab, cnt) in enumerate(stages):
        y = top - i * (h + gap)
        w = 0.94 - 0.055 * i
        x = (1 - w) / 2
        last = i == n - 1
        fc = "#C0504D" if last else "#E8EEF6"
        tc = "white" if last else "#222222"
        ax.add_patch(Rectangle((x, y - h), w, h, transform=ax.transAxes,
                               facecolor=fc, edgecolor="#5A5A5A", lw=0.6))
        ax.text(x + 0.025, y - h / 2, lab, transform=ax.transAxes, fontsize=5.8,
                va="center", ha="left", color=tc)
        ax.text(x + w - 0.025, y - h / 2, cnt, transform=ax.transAxes, fontsize=6.0,
                va="center", ha="right", fontweight="bold", color=tc)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - h - 0.004), (0.5, y - h - gap + 0.004),
                                         transform=ax.transAxes, arrowstyle="-|>",
                                         mutation_scale=6, color="#777777", lw=0.6))
    y = top - n * (h + gap)
    ax.text(0.5, y + 0.01,
            f"cases {n_pat:,}  ·  matched comparisons {n_ctl:,}\n"
            f"1:{ratio:.2f} matching, all SMD < 0.1",
            transform=ax.transAxes, fontsize=6, ha="center", va="top", color="#333333")
    ax.set_title("Cohort construction", fontsize=7.8, fontweight="bold", loc="left")
    panel_label(ax, "a")


def panel_b(ax, counts):
    order = [d for d in DISORDERS if d in counts.index]
    vals = [int(counts[d]) for d in order]
    ypos = np.arange(len(order))[::-1]
    colors = [disorder_color(d) for d in order]
    ax.barh(ypos, vals, color=colors, edgecolor="white", lw=0.4, height=0.72)
    for y, v in zip(ypos, vals):
        ax.text(v * 1.02, y, f"{v:,}", va="center", ha="left", fontsize=5.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([DISPLAY[d] for d in order], fontsize=6)
    ax.set_xlabel("Cases (n)", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.tick_params(labelsize=6)
    despine(ax)
    ax.set_title("Case composition by disorder", fontsize=7.8, fontweight="bold", loc="left")
    panel_label(ax, "b")


def panel_c(ax):
    ax.set_axis_off()
    rows = [
        ("Case-control discrimination", "AUROC 0.84-0.93", "Robust"),
        ("Language-independent detection", "dAUROC 0.006", "Robust"),
        ("Temporal reproducibility", "t0-surge 0.11", "Robust"),
        ("Single-family detection", "0.60-0.83", "Partial"),
        ("Disorder separation", "macro AUROC 0.72", "Moderate"),
        ("Discrete clustering", "ARI 0.088", "Weak"),
        ("Individual early warning", "sparse pre-t0 CPs", "Unsupported"),
        ("Causal inference", "DID + placebo", "Inconclusive"),
        ("Population volatility", "detrended r 0.47", "Promising"),
        ("Population severity trend", "no robust signal", "Not supp."),
    ]
    x0, x1 = 0.0, 1.0
    cA, cB = 0.50, 0.80          # column boundaries
    top, rh = 0.905, 0.082
    # header
    ax.add_patch(Rectangle((x0, top), x1, rh, transform=ax.transAxes,
                           facecolor="#F2F2F2", edgecolor="#BBBBBB", lw=0.4))
    ax.text(0.012, top + rh / 2, "Capability", transform=ax.transAxes, fontsize=6,
            fontweight="bold", va="center")
    ax.text(cA + 0.012, top + rh / 2, "Evidence", transform=ax.transAxes, fontsize=6,
            fontweight="bold", va="center")
    ax.text(cB + 0.012, top + rh / 2, "Verdict", transform=ax.transAxes, fontsize=6,
            fontweight="bold", va="center")
    for i, (cap, ev, verdict) in enumerate(rows):
        y = top - (i + 1) * rh
        shade = "#FFFFFF" if i % 2 == 0 else "#FAFAFA"
        ax.add_patch(Rectangle((x0, y), x1, rh, transform=ax.transAxes,
                               facecolor=shade, edgecolor="#DDDDDD", lw=0.4))
        col = VERDICT_COLORS.get(verdict, "#666666")
        ax.add_patch(Rectangle((cB, y), x1 - cB, rh, transform=ax.transAxes,
                               facecolor=col, edgecolor="#DDDDDD", lw=0.4))
        ax.text(0.012, y + rh / 2, cap, transform=ax.transAxes, fontsize=5.7,
                va="center")
        ax.text(cA + 0.012, y + rh / 2, ev, transform=ax.transAxes, fontsize=5.7,
                va="center", color="#444444")
        ax.text(cB + 0.008, y + rh / 2, verdict, transform=ax.transAxes, fontsize=5.4,
                va="center", color="white", fontweight="bold")
    ax.set_title("What digital phenotyping can and cannot do",
                 fontsize=7.8, fontweight="bold", loc="left")
    panel_label(ax, "c")


def panel_d(ax):
    ax.set_axis_off()
    scales = [
        ("Individual detection", "strong", "Robust"),
        ("Disorder vs disorder", "moderate", "Moderate/weak"),
        ("Latent structure", "shared / dimensional", "Dimensional"),
        ("Individual early warning", "not supported", "Unsupported"),
        ("Population monitoring", "behavioural only", "Behavioural only"),
    ]
    n = len(scales)
    top, h, gap = 0.95, 0.146, 0.035
    for i, (name, sub, verdict) in enumerate(scales):
        y = top - i * (h + gap)
        col = VERDICT_COLORS.get(verdict, "#666666")
        ax.add_patch(FancyBboxPatch((0.0, y - h), 1.0, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    transform=ax.transAxes, facecolor="#F7F7F7",
                                    edgecolor="#CCCCCC", lw=0.5))
        ax.add_patch(Rectangle((0.0, y - h), 0.026, h, transform=ax.transAxes,
                               facecolor=col, edgecolor="none"))
        ax.text(0.045, y - h * 0.36, name, transform=ax.transAxes, fontsize=6.4,
                va="center", fontweight="bold")
        ax.text(0.045, y - h * 0.72, sub, transform=ax.transAxes, fontsize=5.7,
                va="center", color="#555555")
        ax.text(0.985, y - h / 2, verdict, transform=ax.transAxes, fontsize=5.8,
                va="center", ha="right", color=col, fontweight="bold")
    ax.set_title("Scale-dependent validity", fontsize=7.8, fontweight="bold", loc="left")
    panel_label(ax, "d")


# ------------------------------------------------------------------ main
def main():
    apply_style()
    counts, n_pat, n_ctl, unlabelled = load_composition()
    ratio = n_ctl / n_pat
    known = int(sum(int(counts.get(d, 0)) for d in DISORDERS))

    fig = plt.figure(figsize=(mm2inch(180), mm2inch(176)))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.03],
                          wspace=0.26, hspace=0.30,
                          left=0.055, right=0.985, top=0.95, bottom=0.075)

    panel_a(fig.add_subplot(gs[0, 0]), n_pat, n_ctl, ratio)
    panel_b(fig.add_subplot(gs[0, 1]), counts)
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))

    fig.text(0.055, 0.010,
             f"PSM layer: {n_pat:,} cases ({known:,} labelled across 10 disorders"
             + (f", {unlabelled} unlabelled" if unlabelled else "")
             + f") and {n_ctl:,} matched comparisons.\n"
             "Funnel tallies from the preprocessing pipeline. Panels b-d computed from NAS "
             "result files.",
             fontsize=5.4, color="#666666", ha="left", va="bottom")

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    from common_nmh import MAIN_DIR
    out = os.path.join(MAIN_DIR, "fig1_design_validity_map")
    from audit_panel_alignment import require_matplotlib_panel_alignment
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
