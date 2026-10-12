"""
CRITICAL VALIDATION: Window-level regression controlling BOTH event time
(days_from_t0) and calendar time. This is the gold standard to verify whether
the COVID effect survives composition control.

Model (per metric, patient only vs control as baseline):
  Y ~ group + post_covid + group:post_covid + poly(days_from_t0) + calendar_year_linear
Interpretation: group:post_covid = DID-style COVID effect on patient-control gap,
WITH event-time composition controlled.

Uses a subsample (memory); reports coefficient on group x post.
"""
import os
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAJ = "/path/to/project/module2_results/trajectories/user_trajectories.parquet"
COLS = ["user_id_hash", "group", "window_start_ts", "days_from_t0_center",
        "is_reliable", "p_factor_severity_z", "posting_intensity",
        "social_withdrawal_index", "volatility_index"]


def main():
    logger.info("Reading trajectory (sampling for tractability)...")
    pf = pq.ParquetFile(TRAJ)
    chunks = []
    for i, batch in enumerate(pf.iter_batches(batch_size=500_000, columns=COLS)):
        df = batch.to_pandas()
        df = df[df["is_reliable"] == True]
        # subsample 20% for memory
        df = df.sample(frac=0.2, random_state=42) if len(df) > 0 else df
        chunks.append(df)
        if i > 40:
            break
    tr = pd.concat(chunks, ignore_index=True)
    ts = pd.to_datetime(tr["window_start_ts"], unit="s", utc=True)
    tr["year"] = ts.dt.year
    tr["month_float"] = (ts.dt.year - 2010) * 12 + ts.dt.month
    tr = tr[tr["year"].between(2015, 2022)].copy()
    logger.info(f"Analysis sample: {len(tr)} reliable windows 2015-2022")

    from config_module4 import COVID_BREAKPOINT
    bp_y, bp_m = int(COVID_BREAKPOINT[:4]), int(COVID_BREAKPOINT[5:7])
    tr["post"] = (tr["month_float"] >= (bp_y - 2010) * 12 + bp_m).astype(int)
    tr["patient"] = (tr["group"] == "patient").astype(int)
    tr["group_post"] = tr["patient"] * tr["post"]
    # event time polynomial (composition control)
    tr["d2"] = tr["days_from_t0_center"] ** 2
    tr["d3"] = tr["days_from_t0_center"] ** 3

    results = {}
    for metric in ["p_factor_severity_z", "posting_intensity",
                   "social_withdrawal_index", "volatility_index"]:
        d = tr[["patient", "post", "group_post", "days_from_t0_center", "d2", "d3",
                "month_float", metric]].dropna()
        X = d[["patient", "post", "group_post", "days_from_t0_center", "d2", "d3", "month_float"]]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(d[metric], X).fit(cov_type="cluster",
                                             cov_kwds={"groups": tr.loc[d.index, "user_id_hash"]})
            coef = model.params["group_post"]
            se = model.bse["group_post"]
            p = model.pvalues["group_post"]
            results[metric] = {"coef_group_post": float(coef), "se": float(se),
                               "p": float(p), "n": int(len(d)),
                               "month_coef": float(model.params["month_float"])}
            logger.info(f"  {metric}: group*post coef={coef:.5f} (SE={se:.5f}, p={p:.4g}), "
                        f"month coef={model.params['month_float']:.5f}")
        except Exception as e:
            logger.warning(f"  {metric} FAILED: {e}")

    with open("/path/to/project/module4_results/statistics/covid_windowlevel_validation.json", "w") as f:
        import json
        json.dump(results, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
