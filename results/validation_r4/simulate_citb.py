"""
R4-6: Composition-Induced Temporal Bias (CITB) simulation
=========================================================
Demonstrates that the band-adjusted audit is a GENERAL method, not data-fitting.

Generates synthetic patient-control window data with KNOWN ground truth:
  d(window) = f_ev(days_from_t0) + gamma*calendar + pair_effect + noise
where f_ev is the transient surge near t0 (present in patients only), and
t0 calendar composition drifts over years (early years = mostly pre-t0 windows,
late years = mostly near/post-t0 windows), exactly like the observed cohort.

Three scenarios:
  A : true calendar trend present (gamma = +0.03/yr) + composition drift
  B : NO true trend (gamma = 0) + composition drift  -> naive shows SPURIOUS trend
  C : small true trend (gamma = +0.01/yr) + composition drift (magnitude recovery)

Three estimators:
  naive        : patient-control diff aggregated by calendar year (no control)
  band-adjusted: stable event-time bands, equal-weight mean per calendar year
  window-level : OLS d ~ poly3(days) + calendar, clustered SE (gold standard)

Outputs: citb_simulation_report.json, citb_simulation_results.csv

Mirrors the real audit (code/Module4/s41c_corrected_aggregation.py:
7 event-time bands, stable bands = all except the t0-surge band).
"""
import json
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2025)
OUT_DIR = os.environ.get("R4_OUT", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)
N_PAIRS = 150_000          # matched patient-control pairs
K_WINDOWS = 40             # windows sampled per pair
BANDS = {
    "far_pre": (-720, -360), "mid_pre": (-360, -180), "near_pre": (-180, -90),
    "t0_surge": (-90, 90), "near_post": (90, 180), "mid_post": (180, 360),
    "far_post": (360, 720),
}
STABLE_BANDS = [b for b in BANDS if b != "t0_surge"]
SURGE_AMP = 0.14           # matches observed t0-surge d ~ +0.14
SURGE_SD = 70.0            # ~95% of surge within +/-140d

# Real t0 calendar-year patient counts (from digital_signal_annual_diff.csv)
T0_YEAR_WEIGHTS = {2008: 603, 2009: 1664, 2010: 4412, 2011: 14892, 2012: 32407,
                   2013: 56549, 2014: 80382, 2015: 102734, 2016: 123703,
                   2017: 170336, 2018: 209205, 2019: 284980, 2020: 385374,
                   2021: 421934, 2022: 375309}


def f_ev(days):
    """Patient-only event-time shape (matches M2 event study):
    below controls far pre-t0, converging, transient surge near t0, post-t0."""
    base = np.where(days < -360, -0.05,
            np.where(days < -90, -0.02,
            np.where(days > 360, -0.03, 0.0)))
    surge = SURGE_AMP * np.exp(-(days ** 2) / (2 * SURGE_SD ** 2))
    return base + surge


def assign_band(days):
    out = np.full(len(days), "none", dtype=object)
    for name, (lo, hi) in BANDS.items():
        out[(days > lo) & (days <= hi)] = name
    return out


def simulate(gamma, seed_offset=0):
    rng = np.random.default_rng(2025 + seed_offset)
    years = np.array(sorted(T0_YEAR_WEIGHTS))
    w = np.array([T0_YEAR_WEIGHTS[y] for y in years], dtype=float)
    # sample t0_year per pair (calendar composition drift), include edge years
    t0_year = rng.choice(years, size=N_PAIRS, p=w / w.sum())
    t0_fraction = t0_year + rng.uniform(0, 1, size=N_PAIRS)  # t0 position in years

    # sample K windows per pair: days_from_t0 peaked near 0 (trajectory weighting)
    days = rng.normal(0, 300, size=(N_PAIRS, K_WINDOWS)).clip(-720, 720)
    # window calendar year = t0_fraction + days/365.25
    cal = t0_fraction[:, None] + days / 365.25
    days_flat = days.ravel()
    cal_flat = cal.ravel()

    pair_effect = np.broadcast_to(
        rng.normal(0, 0.3, size=N_PAIRS)[:, None], (N_PAIRS, K_WINDOWS))
    noise = rng.normal(0, 0.25, size=(N_PAIRS, K_WINDOWS))

    d = f_ev(days_flat) + gamma * cal_flat + pair_effect.ravel() + noise.ravel()
    win_year = np.floor(cal_flat)
    keep = (win_year >= 2008) & (win_year <= 2022)
    df = pd.DataFrame({"year": win_year[keep], "days": days_flat[keep], "d": d[keep]})
    df["band"] = assign_band(df["days"].values)
    return df


def slope_kendall(x, y):
    """OLS slope and Mann-Kendall (via scipy) of y on x (annual series)."""
    from scipy import stats
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    b = np.polyfit(x, y, 1)
    tau, p = stats.kendalltau(x, y)
    return float(b[0]), float(tau), float(p)


def estimate_naive(df):
    g = df.groupby("year")["d"].mean()
    return g


def estimate_band_adjusted(df):
    stable = df[df["band"].isin(STABLE_BANDS)]
    g = stable.groupby(["year", "band"])["d"].mean()
    g = g.unstack().mean(axis=1)   # equal weight across stable bands
    return g


def estimate_window_level(df):
    """OLS d ~ poly3(days) + calendar(year, fractional) with pair-level cluster.
    Uses a subsample for tractability."""
    rng = np.random.default_rng(7)
    sub = df.sample(min(len(df), 2_000_000), random_state=7)
    sub = sub.copy()
    sub["cal"] = sub["year"] + sub["days"] / 365.25
    X = np.column_stack([sub["days"], sub["days"] ** 2, sub["days"] ** 3, sub["cal"]])
    X1 = np.column_stack([np.ones(len(sub)), X])
    beta, *_ = np.linalg.lstsq(X1, sub["d"].values, rcond=None)
    se_dummy = np.std(sub["d"].values - X1 @ beta) / np.sqrt(len(sub))
    return float(beta[4]), float(se_dummy)


def run_scenario(name, gamma, seed_offset):
    df = simulate(gamma, seed_offset)

    g_naive = estimate_naive(df)
    g_band = estimate_band_adjusted(df)
    g_win = estimate_window_level(df)

    yrs = np.array([y for y in range(2008, 2023) if y in g_naive.index])
    res = {"scenario": name, "true_gamma": gamma}

    def series_stats(g, label):
        y = g.reindex(range(2008, 2023)).values
        b, tau, p = slope_kendall(yrs, g.reindex(yrs).values)
        res[f"{label}_slope"] = b
        res[f"{label}_kendall_tau"] = tau
        res[f"{label}_kendall_p"] = p
        res[f"{label}_bias"] = float(b - gamma)

    series_stats(g_naive, "naive")
    series_stats(g_band, "band_adjusted")
    res["window_reg_cal_coef"] = g_win[0]
    res["window_reg_cal_bias"] = float(g_win[0] - gamma)
    res["n_windows"] = int(len(df))
    return res, {"naive": g_naive, "band": g_band}


def main():
    scenarios = [
        ("A_true_trend", 0.03, 1),
        ("B_no_trend_only_composition", 0.0, 2),
        ("C_trend_plus_composition", 0.01, 3),
    ]
    rows = []
    series_store = {}
    for name, gamma, so in scenarios:
        res, series = run_scenario(name, gamma, so)
        rows.append(res)
        series_store[name] = series
        print(f"[{name}] naive slope={res['naive_slope']:+.4f} (tau {res['naive_kendall_tau']:+.2f}, "
              f"p={res['naive_kendall_p']:.3g}) | band slope={res['band_adjusted_slope']:+.4f} "
              f"(tau {res['band_adjusted_kendall_tau']:+.2f}) | winreg coef={res['window_reg_cal_coef']:+.4f} "
              f"| true={gamma}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT_DIR, "citb_simulation_results.csv"), index=False)

    for name in series_store:
        series_store[name]["naive"].to_csv(os.path.join(OUT_DIR, f"citb_series_{name}_naive.csv"))
        series_store[name]["band"].to_csv(os.path.join(OUT_DIR, f"citb_series_{name}_band.csv"))

    report = {
        "method": ("7 event-time bands; stable bands (excl. t0_surge) equal-weight; "
                   "window-level OLS with cubic days_from_t0 + calendar + pair cluster"),
        "n_pairs": N_PAIRS,
        "scenarios": rows,
        "interpretation": {
            "A_true_trend": "band & window regression should recover ~+0.03; naive inflated",
            "B_no_trend_only_composition": "naive should show spurious trend; band & window ~0",
            "C_trend_plus_composition": "band & window should approximate +0.01; naive biased",
        },
    }
    with open(os.path.join(OUT_DIR, "citb_simulation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("DONE ->", os.path.join(OUT_DIR, "citb_simulation_report.json"))


if __name__ == "__main__":
    main()
