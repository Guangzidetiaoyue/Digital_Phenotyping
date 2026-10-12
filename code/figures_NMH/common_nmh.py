"""Shared helpers for the Nature Mental Health figure set.

Single source of truth for: output directory, unified colour palette, typography,
saving (png/pdf/svg), panel letters. Every NMH figure imports from here so the
whole display set is visually consistent.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- output paths
# figures_nmh/ is a SIBLING of module*_results (project root), per request.
PROJECT_ROOT = "/path/to/project"
NMH_ROOT = os.path.join(PROJECT_ROOT, "figures_nmh")
MAIN_DIR = os.path.join(NMH_ROOT, "main")
ED_DIR = os.path.join(NMH_ROOT, "ed")
for _d in (MAIN_DIR, ED_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- disorders
DISORDERS = ["depression", "bipolar", "anxiety", "ocd", "ptsd", "bpd",
             "eating_disorders", "schizophrenia", "adhd", "autism"]
DISPLAY = {
    "depression": "Depression", "bipolar": "Bipolar", "anxiety": "Anxiety",
    "ocd": "OCD", "ptsd": "PTSD", "bpd": "Borderline PD",
    "eating_disorders": "Eating disorders", "schizophrenia": "Schizophrenia",
    "adhd": "ADHD", "autism": "Autism",
}

# ---------------------------------------------------------------- palette
# Unified, colour-blind-friendly disorder colours (used everywhere).
PALETTE = {
    "depression": "#3B6FB6", "bipolar": "#8C6BB1", "anxiety": "#4C9F70",
    "ocd": "#5FA8C7", "ptsd": "#C0504D", "bpd": "#D98BB5",
    "eating_disorders": "#C7A93B", "schizophrenia": "#7F7F7F",
    "adhd": "#E08A1E", "autism": "#3F7F5F",
}

# Disorder abbreviations (used in plots; key printed in the figure caption).
ABBR = {
    "depression": "MDD", "bipolar": "BD", "anxiety": "ANX", "ocd": "OCD",
    "ptsd": "PTSD", "bpd": "BPD", "eating_disorders": "ED",
    "schizophrenia": "SCZ", "adhd": "ADHD", "autism": "ASD",
}


def abbr(disorder):
    return ABBR.get(disorder, DISPLAY.get(disorder, disorder))


# Elegant, low-saturation palettes (from matplotlib.colors)
from matplotlib.colors import LinearSegmentedColormap
DIV_CMAP = LinearSegmentedColormap.from_list(
    "nmh_div", ["#4A6FA5", "#A7BEDB", "#F5F5F5", "#E0B0AC", "#A85751"])
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "nmh_seq", ["#F5F7FA", "#B4C4D6", "#7B9BC1", "#355C8A"])
SEQ_CAT = ["#DCE0E6", "#B4C4D6", "#8AA6C4", "#5F84AD", "#355C8A"]
ACCENT_RED = "#A85751"
ACCENT_BLUE = "#355C8A"
NEUTRAL = "#8A8A8A"
DARK_BLUE = "#1F3B60"
GREEN = "#4C9F70"
ORANGE = "#E08A1E"
PURPLE = "#8C6BB1"
GOLD = "#C7A93B"

HITOP_SPECTRA = {
    "Internalizing": ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder": ["schizophrenia", "bipolar"],
    "Externalizing": ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental": ["autism"],
}
SPECTRUM_COLORS = {
    "Internalizing": "#C0504D", "Thought Disorder": "#8C6BB1",
    "Externalizing": "#E08A1E", "Detachment/Personality": "#3B6FB6",
    "Neurodevelopmental": "#3F7F5F",
}

# verdict colours for the capability map / scale ladder
VERDICT_COLORS = {
    "Robust": "#1B5E20", "Promising": "#0D47A1", "Partial": "#E65100",
    "Moderate": "#F9A825", "Moderate/weak": "#F9A825",
    "Weak": "#E65100", "Weak/inconclusive": "#E65100", "Inconclusive": "#E65100",
    "Unsupported": "#B71C1C", "Not supported": "#B71C1C", "Not supp.": "#B71C1C",
    "Dimensional": "#0D47A1", "Behavioural only": "#1B5E20",
}


def spectrum_of(disorder):
    for s, ds in HITOP_SPECTRA.items():
        if disorder in ds:
            return s
    return "Unknown"


def disorder_color(disorder):
    return PALETTE.get(disorder, "#808080")


# ---------------------------------------------------------------- typography
def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",     # editable text in SVG
        "pdf.fonttype": 42,         # editable TrueType text in PDF
        "font.size": 7,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.1,
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def mm2inch(mm):
    return mm / 25.4


# ---------------------------------------------------------------- save
def save_fig(fig, name, subdir="main", formats=("png", "pdf", "svg")):
    d = MAIN_DIR if subdir == "main" else ED_DIR
    base = os.path.join(d, name)
    # Mandatory render-time panel-alignment gate (nature-figure skill).
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    except ImportError:
        print("WARN: audit_panel_alignment not on PYTHONPATH; alignment gate skipped")
    paths = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    print("Saved:", ", ".join(paths))
    return paths


def despine(ax, top=True, right=True, left=False, bottom=False):
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def panel_label(ax, letter, dx_pt=-7.0, dy_pt=2.5, fs=9.5):
    """Panel letter at a CONSTANT point offset from the axes top-left.

    Point offsets (not axes fractions) keep labels aligned across panels that
    share a top or left grid boundary; the -9 pt x-offset also clears the
    left-aligned axes title (the nature-figure alignment gate)."""
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx_pt, dy_pt), textcoords="offset points",
                fontsize=fs, fontweight="bold", va="bottom", ha="right",
                annotation_clip=False)


def wrap(text, width=16):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def ring_labels(ax, xs, ys, labels, pad=1.0, min_gap_deg=26.0, color="#222222"):
    """Place labels on a ring around the point cloud with short leaders.

    Keeps labels off the data and off the axes spines (used by the ED scatter panels).
    """
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
        ex, ey = cx + 0.62 * R * np.cos(th), cy + 0.62 * R * np.sin(th)
        ax.plot([x, ex], [y, ey], color="#BBBBBB", lw=0.4, zorder=3)
        ax.annotate(lab, (lx, ly), fontsize=5.0, ha="center", va="center", zorder=6,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(min(xs.min(), cx - R) - 0.25 * R, max(xs.max(), cx + R) + 0.25 * R)
    ax.set_ylim(min(ys.min(), cy - R) - 0.25 * R, max(ys.max(), cy + R) + 0.25 * R)
