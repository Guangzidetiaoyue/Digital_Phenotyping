"""NMH Extended Data Figure ED11 - Feature-level detail (4 panels).
(a) disorder correlation matrix; (b) feature specificity vs differential specificity;
(c) clinical-utility weight sensitivity; (d) transdiagnostic features.
No panel repeats a main or ED panel (different data or view)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, spectrum_of, SPECTRUM_COLORS, DIV_CMAP, ACCENT_BLUE,
                        NEUTRAL)

M1 = "/path/to/project/module1_results/statistics"


def panel_a(ax):
    d = pd.read_csv(os.path.join(M1, "disorder_correlation_matrix.csv"), index_col=0)
    ds = [c for c in DISORDERS if c in d.columns]
    M = d.loc[ds, ds].values.astype(float)
    vmax = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=DIV_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([abbr(x) for x in ds], rotation=90, fontsize=5.4)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([abbr(x) for x in ds], fontsize=5.4)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.5])
    cb = ax.figure.colorbar(ax.images[-1], cax=cax)
    cb.set_label("correlation", fontsize=5.0)
    cb.ax.tick_params(labelsize=5.0)
    ax.tick_params(axis="both", pad=4)
    ax.set_title("Disorder correlation matrix", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(os.path.join(M1, "feature_ranking.csv"), index_col=0)
    x = pd.to_numeric(d["specificity"], errors="coerce")
    y = pd.to_numeric(d["differential_specificity"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=8, color=ACCENT_BLUE, edgecolors="none", alpha=0.6)
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Differential specificity")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Feature specificity", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    j = json.load(open(os.path.join(M1, "cu_weight_sensitivity.json")))
    comps = j["comparisons"]
    rows = [(k, v.get("jaccard_top_k", np.nan)) for k, v in comps.items()]
    rows.sort(key=lambda t: t[1])
    ys = np.arange(len(rows))
    for y, (k, v) in zip(ys, rows):
        ax.scatter([v], [y], s=20, color=ACCENT_BLUE, zorder=4)
        ax.text(v + 0.004, y, f"{v:.2f}", va="center", ha="left", fontsize=5.0,
                color="#333333")
    ax.axvline(j["min_jaccard"], color=NEUTRAL, lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0].replace("_", " ") for r in rows], fontsize=5.0)
    ax.set_xlabel("Top-k overlap (Jaccard)")
    vv = [r[1] for r in rows]
    ax.set_xlim(min(vv) - 0.05, 1.02)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Weight sensitivity", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M1, "transdiagnostic_features.csv"))
    cats = sorted(set(d["category"].dropna()))
    cmap = {c: SPECTRUM_COLORS.get(c, NEUTRAL) for c in cats}
    x = pd.to_numeric(d["n_sig_disorders"], errors="coerce")
    y = pd.to_numeric(d["mean_abs_g_sig"], errors="coerce")
    for c in cats:
        m = (d["category"] == c) & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=10, color=cmap[c], edgecolors="none", alpha=0.7,
                   label=str(c))
    ax.set_xlabel("Significant disorders")
    ax.set_ylabel("Mean |g| (significant)")
    ax.legend(fontsize=5.0, loc="best", frameon=False, handletextpad=0.2,
              labelspacing=0.25)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Transdiagnostic features", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(150)))
    gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.45, left=0.11, right=0.955,
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
    out = os.path.join(ED_DIR, "ed11_feature_detail")
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
