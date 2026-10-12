"""
R4-7: External-validity robustness suite
========================================
Guards the headline cross-correlations (volatility/posting/withdrawal gaps vs
national suicide/MDE statistics) against the classic small-N pitfalls:
common time trend, autocorrelation, single influential years, lag structure.

Inputs (switchable):
  signal_annual : annual patient-control diff (default = available local series;
                  set SIGNAL_PATH to the band-adjusted file when server-accessible)
  external      : data/external/processed/epidemiology_annual.csv

Checks per (signal, external) pair:
  1. raw Pearson (overlap years) + p
  2. first-difference correlation (dx, dy)
  3. detrended correlation (residualize both on year)
  4. Newey-West HAC t/p (OLS external ~ signal, lag=1)
  5. block-permutation p-value (block length 2, 5000 draws)
  6. leave-one-year-out r (min / max / range)
  7. lagged correlation (lag -3..+3, annual)
  8. block-bootstrap 95% CI (block length 2, 2000 draws)

NOTE: the local annual series is the composition-UNadjusted aggregation.
The corrected (band-adjusted) series is generated on the server by
s41c_corrected_aggregation.py (digital_signal_annual_band_adjusted.csv);
the same suite must be re-run on that file before the paper's headline r-values.
Outputs: external_robustness_results.csv, external_robustness_report.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

OUT_DIR = os.environ.get("R4_OUT", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)
# signal path override: pass a path as argv[1]; default = local composition-unadjusted series
DEFAULT_SIGNAL = os.path.normpath(os.path.join(
    OUT_DIR, "..", "module4", "s41-calendar", "20260831-2018-32690",
    "digital_signal_annual_diff.csv"))
EXT_PATH = os.environ.get(
    "R4_EXT",
    os.path.join(OUT_DIR, "..", "..", "data", "external", "processed",
                 "epidemiology_annual.csv"))

SIGNAL_COLS = ["p_factor_severity_z_diff", "posting_intensity_diff",
               "social_withdrawal_index_diff", "volatility_index_diff"]
EXTERNAL_TARGETS = {"suicide_rate": "suicide_rate", "MDE_prevalence": "MDE_prevalence"}
SEED = 2025


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan
    from scipy import stats
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p)


def first_diff_corr(x, y):
    dx, dy = np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float))
    return pearson(dx, dy)


def detrended_corr(x, y):
    yr = np.arange(len(x), dtype=float)
    from scipy import stats
    def resid(v):
        v = np.asarray(v, float)
        b = np.polyfit(yr, v, 1)
        return v - np.polyval(b, yr)
    return pearson(resid(x), resid(y))


def newey_west(x, y):
    """OLS external ~ signal with Newey-West HAC; returns (coef, t, p)."""
    from statsmodels.api import OLS, add_constant
    X = add_constant(np.asarray(x, float))
    m = OLS(np.asarray(y, float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return float(m.params[1]), float(m.tvalues[1]), float(m.pvalues[1])


def block_permutation_p(x, y, block=2, n_perm=5000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    r_obs, _ = pearson(x, y)
    nblocks = int(np.ceil(n / block))
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        idx = []
        order = rng.permutation(nblocks)
        for b in order:
            start = min(b * block, n)
            idx.extend(range(start, min(start + block, n)))
        # rotate within reshaped blocks to break temporal pairing
        idx = np.array(idx)
        r_null, _ = pearson(np.asarray(x)[idx], y)
        nulls[i] = r_null
    return float((np.abs(nulls) >= np.abs(r_obs)).mean())


def leave_one_year_out(x, y, years):
    rs = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        if len(keep) < 5:
            continue
        r, _ = pearson(np.asarray(x)[keep], np.asarray(y)[keep])
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(rs)
    return float(rs.min()), float(rs.max()), float(rs.max() - rs.min())


def lagged(x, y, years, lags=range(-3, 4)):
    out = {}
    for lag in lags:
        yr = np.array(years, float)
        yy = yr + lag
        m = np.isin(yy, yr)
        if m.sum() < 5:
            out[int(lag)] = {"r": np.nan, "p": np.nan, "n": int(m.sum())}
            continue
        j = np.searchsorted(yr, yy[m])
        r, p = pearson(np.asarray(x, float)[m], np.asarray(y, float)[j])
        out[int(lag)] = {"r": r, "p": p, "n": int(m.sum())}
    return out


def block_bootstrap_ci(x, y, block=2, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    nblocks = int(np.ceil(n / block))
    rs = []
    for _ in range(n_boot):
        idx = []
        for _b in range(nblocks):
            s = rng.randint(0, n)
            idx.extend(range(s, min(s + block, n)))
        idx = np.array(idx[:n])
        r, _ = pearson(np.asarray(x)[idx], np.asarray(y)[idx])
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main():
    signal_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIGNAL
    sig = pd.read_csv(os.path.normpath(signal_path))
    ext = pd.read_csv(os.path.normpath(EXT_PATH))

    rows = []
    for scol in SIGNAL_COLS:
        sdf = sig[["year", scol]].dropna()
        for ind_name, ind_col in EXTERNAL_TARGETS.items():
            edf = ext[ext["indicator"] == ind_name][["year", "value"]].rename(
                columns={"value": ind_col})
            m = sdf.merge(edf, on="year", how="inner")
            if len(m) < 6:
                continue
            x = m[scol].values
            y = m[ind_col].values
            years = m["year"].values
            r_raw, p_raw = pearson(x, y)
            r_fd, p_fd = first_diff_corr(x, y)
            r_det, p_det = detrended_corr(x, y)
            nw_coef, nw_t, nw_p = newey_west(x, y)
            perm_p = block_permutation_p(x, y)
            loyo_min, loyo_max, loyo_range = leave_one_year_out(x, y, years)
            lags = lagged(x, y, years)
            ci_lo, ci_hi = block_bootstrap_ci(x, y)
            rows.append({
                "signal": scol, "external": ind_name, "n_years": len(m),
                "raw_r": r_raw, "raw_p": p_raw,
                "firstdiff_r": r_fd, "firstdiff_p": p_fd,
                "detrended_r": r_det, "detrended_p": p_det,
                "neweywest_coef": nw_coef, "neweywest_t": nw_t, "neweywest_p": nw_p,
                "block_permutation_p": perm_p,
                "loyo_min_r": loyo_min, "loyo_max_r": loyo_max, "loyo_range": loyo_range,
                "lag_best_r": max(lags.values(), key=lambda d: abs(d["r"]) if d["r"] == d["r"] else -1)["r"]
                              if any(d["r"] == d["r"] for d in lags.values()) else np.nan,
                "lag_best_lag": next((k for k, d in lags.items() if d["r"] == d["r"] and
                                      abs(d["r"]) == max(abs(dd["r"]) for dd in lags.values() if dd["r"] == dd["r"])), np.nan),
                "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
                "lags": json.dumps(lags),
            })

    res = pd.DataFrame(rows)
    tag = sys.argv[2] if len(sys.argv) > 2 else ("corrected" if len(sys.argv) > 1 else "naive")
    res.to_csv(os.path.join(OUT_DIR, f"external_robustness_results_{tag}.csv"), index=False)

    report = {
        "series": tag,
        "note": ("corrected = band-adjusted (composition-controlled) annual series "
                 "digital_signal_annual_band_adjusted.csv; naive = composition-unadjusted "
                 "digital_signal_annual_diff.csv"),
        "signal_path": str(os.path.normpath(signal_path)),
        "pairs": rows,
    }
    with open(os.path.join(OUT_DIR, f"external_robustness_report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"SERIES={tag}")
    print(res[["signal", "external", "n_years", "raw_r", "raw_p", "firstdiff_r",
               "detrended_r", "neweywest_p", "block_permutation_p", "loyo_range",
               "bootstrap_ci_lo", "bootstrap_ci_hi"]].round(3).to_string(index=False))
    print("DONE ->", os.path.join(OUT_DIR, f"external_robustness_report_{tag}.json"))


if __name__ == "__main__":
    main()
