"""NMH Extended Data Figure ED12 - Auxiliary validation detail (4 panels).
(a) number of diagnoses per user; (b) network density vs graphical-lasso alpha;
(c) digital p-factor vs external epidemiology over time; (d) empirical vs externally
sourced comorbidity. All values are read from the stored result files."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_nmh import (apply_style, mm2inch, despine, panel_label, ED_DIR, DISORDERS,
                        abbr, spectrum_of, SPECTRUM_COLORS, ACCENT_BLUE, NEUTRAL)

M3 = "/path/to/project/module3_results/statistics"
M4 = "/path/to/project/module4_results/statistics"


def panel_a(ax):
    j = json.load(open(os.path.join(M3, "multi_diagnosis_stats.json")))
    dist = j["n_disorders_dist"]
    xs = sorted(int(k) for k in dist.keys())
    ys = [dist[str(k)] for k in xs]
    ax.bar(xs, ys, color=ACCENT_BLUE, edgecolor="none")
    ax.set_xlabel("Diagnoses per user")
    ax.set_ylabel("Users")
    ax.set_xticks(xs)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Multi-diagnosis users", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    j = json.load(open(os.path.join(M3, "network_alpha_sensitivity.json")))
    for d in DISORDERS:
        if d not in j:
            continue
        alphas = sorted(float(a) for a in j[d].keys())
        dens = [j[d][f"{a:.1f}"]["density"] if f"{a:.1f}" in j[d]
                else j[d][str(a)]["density"] for a in alphas]
        ax.plot(alphas, dens, marker="o", ms=2.0, lw=0.9,
                color=SPECTRUM_COLORS[spectrum_of(d)], alpha=0.85)
    ax.set_xlabel("Graphical-lasso alpha")
    ax.set_ylabel("Network density")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Network density vs alpha", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    d = pd.read_csv(os.path.join(M4, "trend_comparison_external.csv")).dropna(
        subset=["digital_pfactor_diff", "suicide_rate"])
    yr = d["year"].values
    def z(v):
        v = np.asarray(v, float)
        return (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)
    ax.plot(yr, z(d["digital_pfactor_diff"]), color=ACCENT_BLUE, lw=1.1,
            label="p-factor gap")
    ax.plot(yr, z(d["suicide_rate"]), color=NEUTRAL, lw=1.1, label="suicide rate")
    if "MDE_pct" in d and d["MDE_pct"].notna().any():
        md = d.dropna(subset=["MDE_pct"])
        ax.plot(md["year"], z(md["MDE_pct"]), color="#B4C4D6", lw=1.0, label="MDE")
    ax.axhline(0, color="#DDDDDD", lw=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardised")
    ax.legend(fontsize=5.0, loc="best", frameon=False)
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Digital vs external trends", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    j = json.load(open(os.path.join(M3, "empirical_comorbidity_mantel.json")))
    ej, ext = j["empirical_jaccard"], j["external_sourced_comorbidity"]
    dis = j.get("disorders", list(ej.keys()) if isinstance(ej, dict) else [])
    xs, ys = [], []
    nested = (isinstance(ej, dict) and isinstance(ext, dict)
              and len(ej) > 0 and isinstance(list(ej.values())[0], dict))
    if nested:
        keys = [d for d in dis if d in ej and d in ext]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                try:
                    ev, jv = float(ext[a][b]), float(ej[a][b])
                except Exception:
                    continue
                xs.append(ev)
                ys.append(jv)
    else:
        if isinstance(ej, dict) and isinstance(ext, dict):
            for k in ej.keys():
                if k in ext:
                    xs.append(float(ext[k]))
                    ys.append(float(ej[k]))
        else:
            xs = [float(v) for v in ext]
            ys = [float(v) for v in ej]
    ax.scatter(xs, ys, s=22, color=ACCENT_BLUE, edgecolors="white", lw=0.5, zorder=4)
    ax.set_xlabel("Externally sourced comorbidity")
    ax.set_ylabel("Empirical Jaccard")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title("Empirical vs external", loc="left", fontsize=7.8)
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
    out = os.path.join(ED_DIR, "ed12_auxiliary_validation")
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
