"""NMH Extended Data Figure ED8 - Network and intermediate-state detail (4 panels).
(a) edge sharing; (b) intermediate-state scatter; (c) network topology; (d) empirical
comorbidity (Jaccard) matrix."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, SEQ_CMAP, ACCENT_BLUE, ACCENT_RED, NEUTRAL)

M3 = "/path/to/project/module3_results/statistics"


def panel_a(ax):
    d = pd.read_csv(os.path.join(M3, "network_edge_consistency.csv"))
    v = pd.to_numeric(d["n_disorders"], errors="coerce").dropna().astype(int).values
    ax.hist(v, bins=np.arange(0, v.max() + 2) - 0.5, color=ACCENT_BLUE,
            edgecolor="white", lw=0.4)
    ax.set_xlabel("Disorders sharing the edge")
    ax.set_ylabel("Edges")
    ax.set_xticks(range(0, v.max() + 1))
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Edge sharing", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(os.path.join(M3, "comorbidity_intermediate_results.csv"))
    x = pd.to_numeric(d["frac_intermediate"], errors="coerce").values.astype(float)
    y = pd.to_numeric(d["res_ratio"], errors="coerce").values.astype(float)
    sig = pd.to_numeric(d["p_binom_fdr"], errors="coerce").values < 0.05
    ax.scatter(x[sig], y[sig], s=22, color=ACCENT_RED, edgecolors="white", lw=0.5,
               zorder=4, label="FDR < 0.05")
    ax.scatter(x[~sig], y[~sig], s=22, color=NEUTRAL, edgecolors="white", lw=0.5,
               zorder=4, label="n.s.")
    ax.axhline(1.0, color="#BBBBBB", lw=0.6, ls=":")
    ax.set_xlabel("Intermediate-state fraction")
    ax.set_ylabel("Residual ratio")
    ax.legend(fontsize=5.0, loc="best", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", handletextpad=0.3, borderpad=0.3)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Intermediate states", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M3, "disorder_network_topology.csv"))
    ds = [x for x in DISORDERS if x in set(d["disorder"])]
    d = d.set_index("disorder").loc[ds].sort_values("n_edges")
    ds = list(d.index)
    ys = np.arange(len(ds))
    for y, x in zip(ys, ds):
        ax.scatter([d.loc[x, "n_edges"]], [y], s=20, color=ACCENT_BLUE,
                   edgecolors="none", zorder=4)
        ax.text(d.loc[x, "n_edges"] + 12, y, f"{d.loc[x, 'density']:.2f}", va="center",
                ha="left", fontsize=5.0, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(x) for x in ds], fontsize=5.6)
    ax.set_xlabel("Network edges (annotated: density)")
    ax.set_xlim(0, d["n_edges"].max() * 1.25)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Network topology", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(os.path.join(M3, "comorbidity_empirical_jaccard.csv"), index_col=0)
    cols = [c for c in DISORDERS if c in d.columns]
    M = d.loc[cols, cols].values.astype(float)
    ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([abbr(c) for c in cols], rotation=90, fontsize=5.4)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([abbr(c) for c in cols], fontsize=5.4)
    vis = np.zeros_like(M, bool)
    for i in range(len(cols)):
        for j in range(len(cols)):
            vis[i, j] = i != j
    ax.imshow(np.ma.masked_where(~vis, M), cmap=SEQ_CMAP, aspect="auto", vmin=0,
              vmax=np.nanmax(M))
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.5])
    cb = ax.figure.colorbar(ax.images[-1], cax=cax)
    cb.set_label("Jaccard", fontsize=5.0)
    cb.ax.tick_params(labelsize=5.0)
    ax.tick_params(axis="both", pad=4)
    ax.set_title("Empirical comorbidity", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed8_network_detail")
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
