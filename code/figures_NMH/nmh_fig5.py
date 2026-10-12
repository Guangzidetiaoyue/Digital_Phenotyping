"""NMH Figure 5 - Population behavioural responses to societal perturbations.

Message: at the population scale the digital phenotype tracks societal shocks only
weakly, and only behavioural volatility carries a usable group-level signal.

Panels (role-distinct chart types):
  a  Line       : monthly patient-minus-comparison gap (dots = monthly, line = 6-mo mean)
  b  Forest     : COVID-19 ITS level shift per disorder (p-factor vs volatility)
  c  Forest     : pooled event-study DID for five societal events
  d  Heatmap    : COVID-19 trend change, signals x disorders (FDR stars)
  e  Scatter    : per-disorder long-term trend vs COVID level shift
  f  Ridgeline  : per-disorder volatility series with COVID shading
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.dates import YearLocator, DateFormatter

from common_nmh import (apply_style, mm2inch, despine, panel_label, MAIN_DIR,
                        DISORDERS, abbr, spectrum_of, SPECTRUM_COLORS, DIV_CMAP,
                        SEQ_CMAP, ACCENT_BLUE, ACCENT_RED, GREEN, ORANGE)

B = "/path/to/project"
D4 = os.path.join(B, "module4_results", "data")
S4 = os.path.join(B, "module4_results", "statistics")
MONTHLY = os.path.join(D4, "digital_signal_monthly_pooled.csv")
BYDIS = os.path.join(D4, "digital_signal_monthly_by_disorder.csv")
ITS = os.path.join(S4, "covid_its_corrected_results.csv")
EVENTS = os.path.join(S4, "event_study_final.json")
TREND = os.path.join(S4, "trend_by_disorder.csv")

METRICS = [("p_factor_severity_z_diff", "p-factor", "#355C8A"),
           ("posting_intensity_diff", "posting", "#A85751"),
           ("social_withdrawal_index_diff", "withdrawal", GREEN),
           ("volatility_index_diff", "volatility", ORANGE)]
COL = {m: c for m, _, c in METRICS}
COV_START = "2020-03"
P0, P1 = pd.Period("2015-01", "M"), pd.Period("2022-12", "M")


def _pooled_gaps():
    d = pd.read_csv(MONTHLY)
    d = d[d["group"].isin(["patient", "control"])].copy()
    piv = d.pivot_table(index="period", columns="group",
                        values=["p_factor_severity_z_mean", "posting_intensity_mean",
                                "social_withdrawal_index_mean", "volatility_index_mean"],
                        aggfunc="mean").sort_index()
    idx = pd.PeriodIndex(piv.index, freq="M")
    piv = piv[(idx >= P0) & (idx <= P1)]
    return piv, pd.PeriodIndex(piv.index, freq="M")


def panel_a(ax):
    piv, idx = _pooled_gaps()
    x = idx.to_timestamp()
    ends = []
    for base, lab, c in [("p_factor_severity_z_mean", "p-factor", COL["p_factor_severity_z_diff"]),
                         ("posting_intensity_mean", "posting", COL["posting_intensity_diff"]),
                         ("social_withdrawal_index_mean", "withdrawal", COL["social_withdrawal_index_diff"]),
                         ("volatility_index_mean", "volatility", COL["volatility_index_diff"])]:
        gap = (piv[(base, "patient")] - piv[(base, "control")]).values
        z = (gap - np.nanmean(gap)) / (np.nanstd(gap) or 1.0)
        ax.scatter(x, z, s=5.0, color=c, alpha=0.35, lw=0, zorder=2)
        rm = pd.Series(z).rolling(6, min_periods=3, center=True).mean().values
        ax.plot(x, rm, color=c, lw=1.7, zorder=3)
        finite = rm[np.isfinite(rm)]
        ends.append([lab, float(finite[-1]) if len(finite) else 0.0, c])
    cov = pd.Timestamp(COV_START)
    ax.axvspan(cov, x.max(), color="#D9D9D9", alpha=0.30, lw=0)
    ax.axvline(cov, color="#888888", lw=0.7, ls="--")
    ax.axhline(0, color="#BBBBBB", lw=0.6)
    lo, hi = ax.get_ylim()
    ends.sort(key=lambda t: t[1])
    step = 0.10 * (hi - lo)
    prev = -1e9
    for e in ends:
        if e[1] < prev + step:
            e[1] = prev + step
        prev = e[1]
    for lab, yv, c in ends:
        ax.annotate(lab, (1.010, yv), xycoords=("axes fraction", "data"), fontsize=5.0,
                    color=c, ha="left", va="center", annotation_clip=False)
    ax.set_ylabel("Patient - comparison gap (z)")
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.tick_params(axis="both", pad=6, labelsize=5.6)
    despine(ax)
    ax.set_title("Monthly digital-signal gap", loc="left", fontsize=7.8)
    panel_label(ax, "a")


def panel_b(ax):
    d = pd.read_csv(ITS)
    sub = d[d["metric"].isin(["p_factor_severity_z_diff", "volatility_index_diff"])]
    order = (sub[sub["metric"] == "p_factor_severity_z_diff"]
             .set_index("disorder")["b_post_shift"].sort_values())
    ds = [x for x in order.index]
    ys = np.arange(len(ds))
    for metric, lab, c in [("p_factor_severity_z_diff", "p-factor", ACCENT_BLUE),
                           ("volatility_index_diff", "volatility", COL["volatility_index_diff"])]:
        s = sub[sub["metric"] == metric].set_index("disorder")
        off = -0.16 if metric == "p_factor_severity_z_diff" else 0.16
        for y, dd in zip(ys, ds):
            b = s.loc[dd, "b_post_shift"]
            se = s.loc[dd, "se_post_shift"]
            sig = s.loc[dd, "p_post_shift_fdr"] < 0.05
            ax.plot([b - 1.96 * se, b + 1.96 * se], [y + off, y + off], color=c, lw=1.0)
            ax.scatter([b], [y + off], s=16, color=c,
                       edgecolors="white" if sig else "none",
                       linewidths=0.6, zorder=4)
        ax.plot([], [], color=c, lw=1.2, label=lab)
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([abbr(x) for x in ds], fontsize=5.6)
    ax.set_ylim(-1.1, len(ds) - 0.4)
    ax.set_xlabel("COVID-19 level shift (95% CI)")
    ax.legend(fontsize=5.0, loc="lower right", frameon=True, framealpha=1.0,
              edgecolor="#CCCCCC", handlelength=1.2, handletextpad=0.3, borderpad=0.3)
    ax.margins(x=0.08)
    ax.tick_params(axis="both", pad=5)
    despine(ax)
    ax.set_title("COVID-19 onset: level shift", loc="left", fontsize=7.8)
    panel_label(ax, "b")


def panel_c(ax):
    ev = json.load(open(EVENTS))["events"]
    short = {"E1_election_2016": "Election 2016", "E2_celebrity_2018": "Celeb. suicides",
             "E3_covid_2020": "COVID-19 onset", "E4_blm_2020": "BLM protests",
             "E5_capitol_2021": "Capitol attack"}
    rows = []
    for k in ev.keys():
        v = ev[k]["p_factor_severity_z"]
        rows.append((short.get(k, ev[k]["label"]), v["coef_group_event"], v["se"], v["p"]))
    rows = rows[::-1]
    ys = np.arange(len(rows))
    for y, (lab, b, se, p) in zip(ys, rows):
        c = ACCENT_BLUE if p < 0.05 else "#BBBBBB"
        ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y], color=c, lw=1.2, zorder=3)
        ax.scatter([b], [y], s=22, color=c, edgecolors="white", lw=0.6, zorder=4)
        ax.text(b + 1.96 * se + 0.0015, y, f"p={p:.2f}", va="center", ha="left",
                fontsize=5.0, color="#333333")
    ax.axvline(0, color="#999999", lw=0.7, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.4)
    ax.set_xlabel("Pooled DID (p-factor), 95% CI")
    ax.margins(x=0.16)
    ax.tick_params(axis="both", pad=5)
    despine(ax)
    ax.set_title("Societal events", loc="left", fontsize=7.8)
    panel_label(ax, "c")


def panel_d(ax):
    d = pd.read_csv(ITS)
    mets = [m for m, _, _ in METRICS]
    M = np.full((len(DISORDERS), len(mets)), np.nan)
    S = np.zeros_like(M, bool)
    for i, dd in enumerate(DISORDERS):
        for j, m in enumerate(mets):
            r = d[(d["disorder"] == dd) & (d["metric"] == m)]
            if len(r):
                M[i, j] = r["b_post_trend"].values[0]
                S[i, j] = r["p_post_trend_fdr"].values[0] < 0.05
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap=DIV_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if S[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, color="#222222")
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([lab for _, lab, _ in METRICS], rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(DISORDERS)))
    ax.set_yticklabels([abbr(x) for x in DISORDERS], fontsize=5.6)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 0.5])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("trend change", fontsize=5.0)
    cb.ax.tick_params(labelsize=5.0)
    ax.set_title("COVID-19 trend change\n* (FDR < 0.05)", loc="left", fontsize=7.8)
    panel_label(ax, "d")


def _ring_labels(ax, xs, ys, labels, pad=1.5, min_gap_deg=22.0):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    a = ang[order].copy()
    g = np.deg2rad(min_gap_deg)
    for k in range(1, n):
        if a[k] - a[k - 1] < g:
            a[k] = a[k - 1] + g
    ang[order] = a
    R = pad * max((xs.max() - xs.min()) or 1.0, (ys.max() - ys.min()) or 1.0)
    for x, y, th, lab in zip(xs, ys, ang, labels):
        lx, ly = cx + R * np.cos(th), cy + R * np.sin(th)
        ex, ey = cx + 0.85 * R * np.cos(th), cy + 0.85 * R * np.sin(th)
        ax.plot([x, ex], [y, ey], color="#BBBBBB", lw=0.4, zorder=3)
        ax.annotate(lab, (lx, ly), fontsize=5.0, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(min(xs.min(), cx - R) - 0.30 * R, max(xs.max(), cx + R) + 0.30 * R)
    ax.set_ylim(min(ys.min(), cy - R) - 0.30 * R, max(ys.max(), cy + R) + 0.30 * R)


def panel_e(ax):
    tr = pd.read_csv(TREND).set_index("disorder")
    d = pd.read_csv(ITS)
    pf = d[d["metric"] == "p_factor_severity_z_diff"].set_index("disorder")
    ds = [x for x in DISORDERS if x in tr.index and x in pf.index]
    xr = np.array([pf.loc[dd, "b_post_shift"] for dd in ds])
    x = (xr - xr.mean()) / (xr.std() or 1.0)
    yr = np.array([tr.loc[dd, "tau"] for dd in ds])
    y = (yr - yr.mean()) / (yr.std() or 1.0)
    ax.scatter(x, y, s=26, c=[SPECTRUM_COLORS[spectrum_of(dd)] for dd in ds],
               edgecolors="white", lw=0.6, zorder=4)
    _ring_labels(ax, x, y, [abbr(dd) for dd in ds], pad=0.95, min_gap_deg=36)
    r = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("COVID shift (z, p-factor)")
    ax.set_ylabel("tau (z)")
    ax.tick_params(axis="both", pad=5, labelsize=5.6)
    despine(ax)
    ax.set_title(f"Trend vs COVID shift (r={r:+.2f})", loc="left", fontsize=7.8)
    panel_label(ax, "e")


def panel_f(ax):
    d = pd.read_csv(BYDIS)
    d = d[d["disorder_group"].isin(DISORDERS)].copy()
    ds = [dd for dd in DISORDERS if dd in set(d["disorder_group"])]
    step = 3.2
    for k, dd in enumerate(ds):
        s = d[d["disorder_group"] == dd].groupby("period")["volatility_index"].mean()
        pi = pd.PeriodIndex(s.index, freq="M")
        s = s[(pi >= P0) & (pi <= P1)].sort_index()
        pi = pd.PeriodIndex(s.index, freq="M")
        x = pi.to_timestamp()
        v = pd.to_numeric(s.values, errors="coerce").astype(float)
        v = (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)
        v = pd.Series(v).rolling(3, min_periods=1, center=True).mean().values
        base = k * step
        c = SPECTRUM_COLORS[spectrum_of(dd)]
        ax.plot(x, v + base, color=c, lw=0.7, alpha=0.85)
        ax.axhline(base, color="#EFEFEF", lw=0.4, zorder=0)
        ax.annotate(abbr(dd), (x.min(), base), xytext=(-2, 0), textcoords="offset points",
                    fontsize=5.0, ha="right", va="center", color=c, annotation_clip=False)
    ax.axvspan(pd.Timestamp(COV_START), pd.Timestamp("2022-12-01"), color="#D9D9D9",
               alpha=0.30, lw=0, zorder=0)
    ax.axvline(pd.Timestamp(COV_START), color="#888888", lw=0.7, ls="--", zorder=2)
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-01"))
    ax.xaxis.set_major_locator(YearLocator(2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.tick_params(axis="x", pad=5, labelsize=5.6)
    despine(ax, left=True)
    ax.set_title("Volatility by disorder (z, offset)", loc="left", fontsize=7.8)
    panel_label(ax, "f")


def main():
    apply_style()
    fig = plt.figure(figsize=(mm2inch(180), mm2inch(170)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.78, hspace=0.42,
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
    out = os.path.join(MAIN_DIR, "fig5_population_perturbations")
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
