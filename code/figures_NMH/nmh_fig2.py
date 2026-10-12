"""NMH Figure 2 — A reproducible, multidimensional, diagnosis-language-independent atlas.

Core conclusion: across ten disorders the digital phenotype shows systematic,
multidimensional effects, robust discrimination, and little dependence on
diagnosis-triggering language.

Panels (one inferential role each)
  a  Feature–disorder effect sizes (Hedges' g)
  b  Effect magnitude by feature dimension
  c  Statistical significance vs effect size
  d  Case vs comparison discrimination (AUROC)
  e  Leakage A: removing diagnosis-triggered posts
  f  Contribution of each feature dimension
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        DISORDERS, DISPLAY, abbr, DIV_CMAP, SEQ_CAT, ACCENT_RED,
                        ACCENT_BLUE, NEUTRAL)

B = "/path/to/project"
M1 = os.path.join(B, "module1_results")
MS = os.path.join(M1, "statistics", "effect_size_matrix_clean.csv")
SF = os.path.join(M1, "statistics", "effect_sizes_full.csv")
META = os.path.join(M1, "features", "feature_metadata.json")
AUC = os.path.join(M1, "statistics", "discrimination_disorder_vs_control.csv")
ABL = os.path.join(M1, "statistics", "dimension_ablation.csv")
LEAK = os.path.join(B, "module4_results", "validation_r4", "r4_leakage_a_results.csv")

SHORT = {"depression": "Depression", "bipolar": "Bipolar", "anxiety": "Anxiety",
         "ocd": "OCD", "ptsd": "PTSD", "bpd": "BPD", "eating_disorders": "Eating",
         "schizophrenia": "Schizophr.", "adhd": "ADHD", "autism": "Autism"}

DIM_DISPLAY = {
    "affective": "Affect", "pronoun": "Pronoun", "cognitive": "Cognition",
    "temporal_linguistic": "Temp. lang.", "risk": "Risk", "behavioral": "Behaviour",
    "readability": "Readability", "post_length": "Post length", "temporal": "Temporal",
    "social": "Social", "community": "Community", "engagement": "Engagement",
    "circadian": "Circadian", "activity": "Activity",
}
CAT_ORDER = ["negligible", "minimal", "small", "medium", "large"]
CAT_COLOR = dict(zip(CAT_ORDER, SEQ_CAT))
CAT_LABEL = {"negligible": "Negligible", "minimal": "Minimal", "small": "Small",
             "medium": "Medium", "large": "Large"}


def content_dims(meta):
    return [d for d in meta.keys() if "transform" not in d.lower()]


# ------------------------------------------------------------------ panels
def panel_a(ax, mat, meta):
    dim_order = list(meta.keys())
    feats = [f for d in dim_order for f in meta[d] if f in mat.index]
    data = mat.loc[feats, DISORDERS].values.astype(float)
    vmax = 1.4
    im = ax.imshow(np.clip(data, -vmax + 1e-6, vmax - 1e-6), aspect="auto",
                   cmap=DIV_CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                   interpolation="nearest")
    ax.set_xticks(range(len(DISORDERS)))
    ax.set_xticklabels([abbr(d) for d in DISORDERS], rotation=90, fontsize=6.4)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    y = 0
    for d in dim_order:
        n = sum(1 for f in meta[d] if f in mat.index)
        if n == 0:
            continue
        ax.add_patch(Rectangle((-0.8, y - 0.5), 0.8, n, facecolor=NEUTRAL,
                               edgecolor="none", clip_on=False))
        ax.text(-1.05, y + n / 2 - 0.5, DIM_DISPLAY.get(d, d), va="center", ha="right",
                fontsize=6.1, clip_on=False)
        ax.axhline(y - 0.5, color="white", lw=0.4)
        y += n
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(DISORDERS) - 0.5)
    ax.set_ylim(len(feats) - 0.5, -0.5)
    ax.set_title("Effect-size atlas (Hedges' g)", loc="left", fontsize=7.8)
    panel_label(ax, "a")
    return im


def panel_b(ax, sf, feat_to_dim, dims):
    rows = []
    for _, r in sf.iterrows():
        d = feat_to_dim.get(r["feature"])
        if d is None or "transform" in str(d).lower():
            continue
        rows.append((d, abs(float(r["hedges_g"]))))
    df = pd.DataFrame(rows, columns=["dim", "absg"])
    order = [d for d in df.groupby("dim")["absg"].median().sort_values(ascending=False).index
             if d in dims]
    data = [df.loc[df["dim"] == d, "absg"].values for d in order]
    ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
               medianprops=dict(color=ACCENT_BLUE, lw=1.1),
               whiskerprops=dict(color=NEUTRAL, lw=0.8),
               capprops=dict(color=NEUTRAL, lw=0.8),
               boxprops=dict(facecolor="#BFCFE2", edgecolor="#6E86A6", lw=0.6))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([DIM_DISPLAY.get(d, d) for d in order], fontsize=6.4)
    ax.set_xlabel("|Hedges' g|")
    q1 = df.groupby("dim")["absg"].quantile(0.25)
    q3 = df.groupby("dim")["absg"].quantile(0.75)
    whis = float((q3 + 1.5 * (q3 - q1)).max())     # upper whisker across dimensions
    xmax = 1.8 if whis <= 1.8 else float(np.ceil(whis * 20) / 20)
    ax.set_xlim(0, xmax)
    despine(ax)
    ax.set_title("Magnitude by dimension", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax, sf):
    d = sf.copy()
    if "is_selection_artifact" in d.columns:
        d = d[~d["is_selection_artifact"].astype(bool)]
    y = -np.log10(d["p_fdr_global"].clip(lower=1e-300).astype(float).values)
    x = d["hedges_g"].astype(float).values
    cats = d.get("effect_category", pd.Series(["small"] * len(d))).fillna("small").values
    for c in CAT_ORDER:
        m = cats == c
        if m.any():
            ax.scatter(x[m], y[m], s=7, c=CAT_COLOR[c], edgecolors="none",
                       alpha=0.8, label=CAT_LABEL[c])
    ax.axvline(0, color="#AAAAAA", lw=0.6)
    ax.legend(fontsize=6.2, loc="upper left", title="Effect size", title_fontsize=6.2,
              handletextpad=0.35, labelspacing=0.32, borderaxespad=0.35,
              frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)
    ax.set_xlabel("Hedges' g")
    ax.set_ylabel("−log10(FDR p)")
    xm = float(np.ceil(np.nanmax(np.abs(x)) * 10) / 10)
    ax.set_xlim(-xm * 1.45, xm * 1.03)
    despine(ax)
    ax.set_title("Significance vs effect size", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax, auc):
    a = auc.copy(); a.index = [str(i) for i in a.index]
    order = [d for d in DISORDERS if d in a.index]
    vals = [float(a.loc[d, "logistic_l2_auroc"]) for d in order]
    errs = [float(a.loc[d, "logistic_l2_auroc_std"]) for d in order]
    ypos = np.arange(len(order))[::-1]
    ax.errorbar(vals, ypos, xerr=errs, fmt="o", ms=5, lw=1.1, capsize=2.5,
                color=ACCENT_BLUE, ecolor="#A9C0DC", mec="white", mew=0.6)
    for y, v in zip(ypos, vals):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", ha="left", fontsize=6.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([abbr(d) for d in order], fontsize=6.6)
    ax.set_xlim(0.82, 0.952)
    ax.set_xlabel("AUROC (case vs comparison)")
    despine(ax)
    ax.set_title("Discrimination (AUROC)", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def panel_e(ax, leak):
    d = leak.copy().sort_values("auroc_original", ascending=False)
    ypos = np.arange(len(d))[::-1]
    no = d["auroc_nodiag"].astype(float).values
    ori = d["auroc_original"].astype(float).values
    lo = d["auroc_nodiag_min"].astype(float).values
    hi = d["auroc_nodiag_max"].astype(float).values
    for y, a, b in zip(ypos, no, ori):
        ax.plot([a, b], [y, y], color="#C9C9C9", lw=1.2, zorder=1)
    ax.hlines(ypos, lo, hi, color="#A9C0DC", lw=0.9, alpha=0.8, zorder=1)
    ax.scatter(no, ypos, s=16, facecolors="white", edgecolors=ACCENT_BLUE, lw=1.0,
               zorder=3, label="No diagnosis posts")
    ax.scatter(ori, ypos, s=17, color=ACCENT_RED, zorder=3, label="Atlas")
    ax.set_yticks(ypos)
    ax.set_yticklabels([abbr(x) for x in d["disorder"]], fontsize=6.6)
    ax.set_xlabel("AUROC")
    ax.set_xlim(min(lo) - 0.012, max(ori) + 0.014)
    ax.legend(fontsize=6.2, loc="upper left", handletextpad=0.4, borderaxespad=0.35,
              frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)
    despine(ax)
    ax.set_title("Leakage: no diagnosis posts", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax, abl, dims, meta):
    a = abl.copy(); a.index = [str(i) for i in a.index]
    cols = [c for c in a.columns if c.startswith("delta_")
            and c.replace("delta_", "") in dims]
    stats = {c.replace("delta_", ""): (float(a[c].mean()), float(a[c].std(ddof=1)))
             for c in cols}
    s = pd.Series({k: v[0] for k, v in stats.items()}).sort_values()
    order = list(s.index)
    ypos = np.arange(len(order))
    vals = s.values
    errs = np.array([stats[k][1] for k in order])
    colors = [ACCENT_RED if v > 0.006 else ("#BFCFE2" if v > 0 else "#E2E2E2") for v in vals]
    ax.barh(ypos, vals, xerr=errs, color=colors, edgecolor="#6E86A6", lw=0.4, height=0.66,
            error_kw=dict(ecolor="#7A7A7A", elinewidth=0.6, capsize=1.6))
    for y, v, e in zip(ypos, vals, errs):
        ax.text(v + e + 0.004, y, f"{v:+.3f}", va="center", ha="left", fontsize=5.4,
                color="#444444")
    ax.set_yticks(ypos)
    ax.set_yticklabels([DIM_DISPLAY.get(d, d) for d in order], fontsize=6.4)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_xlim(min(vals - errs) * 1.35, max(vals + errs) * 1.55)
    ax.set_xlabel("Mean ΔAUROC when dimension removed (mean ± SD across 10 disorders)")
    despine(ax)
    ax.set_title("Dimension contribution", loc="left", fontsize=7.8)

    # inset (fills the empty right space): dimension size vs contribution
    ins = ax.inset_axes([0.68, 0.08, 0.30, 0.66])
    xs = np.array([len(meta.get(d, [])) for d in order], dtype=float)
    ys = np.array([stats[d][0] for d in order])
    ins.axhline(0, color="#CCCCCC", lw=0.5)
    ins.scatter(xs, ys, s=15, color=ACCENT_BLUE, edgecolors="white", lw=0.5, zorder=3)
    ins.set_title("n features vs contribution", fontsize=5.6, loc="left", pad=2)
    ins.set_ylabel("Mean ΔAUROC", fontsize=5.6)
    ins.tick_params(labelsize=5.0)
    despine(ins)
    panel_label(ax, "f")


def main():
    apply_style()
    mat = pd.read_csv(MS, index_col=0)
    sf = pd.read_csv(SF)
    meta = json.load(open(META))
    auc = pd.read_csv(AUC, index_col=0)
    abl = pd.read_csv(ABL, index_col=0)
    leak = pd.read_csv(LEAK)
    feat_to_dim = {f: d for d, fs in meta.items() for f in fs}
    dims = content_dims(meta)

    # final width: 180 mm (Nature double column)
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(235)))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.12, 1.0, 1.0],
                          height_ratios=[1.0, 1.0, 1.0],
                          wspace=0.60, hspace=0.45,
                          left=0.125, right=0.975, top=0.955, bottom=0.05)
    panel_a(fig.add_subplot(gs[:, 0]), mat, meta)
    panel_b(fig.add_subplot(gs[0, 1]), sf, feat_to_dim, dims)
    panel_c(fig.add_subplot(gs[0, 2]), sf)
    panel_d(fig.add_subplot(gs[1, 1]), auc)
    panel_e(fig.add_subplot(gs[1, 2]), leak)
    panel_f(fig.add_subplot(gs[2, 1:3]), abl, dims, meta)

    # NOTE: disorder abbreviations are used in the axes; the abbreviation key is
    # kept OUT of the figure (to avoid cluttering it) and will be given in the
    # figure caption / main text.

    # --- explicit publication export (editable text; vector + raster; alignment gate) ---
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "svg.fonttype": "none",       # editable text in SVG
        "pdf.fonttype": 42,           # editable TrueType text in PDF
    })
    from audit_panel_alignment import require_matplotlib_panel_alignment
    out = os.path.join(MAIN_DIR, "fig2_atlas_detectability")
    require_matplotlib_panel_alignment(
        fig, json_out=out + ".alignment.json", overlay_svg=out + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".svg", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Exported:", out + ".{png,pdf,svg,tiff}")
    print("n_features(heatmap)=", mat.shape[0], "n_disorders=", mat.shape[1],
          "n_effect_tests=", int((~sf["is_selection_artifact"].astype(bool)).sum()))


if __name__ == "__main__":
    main()
