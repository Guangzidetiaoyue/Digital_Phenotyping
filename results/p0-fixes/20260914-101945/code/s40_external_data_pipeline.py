"""
M4 External Data Pipeline (v1)
==============================
Convert raw external data -> unified processed format for RQ4 analysis.

Outputs (processed/):
  - epidemiology_annual.csv   : year,indicator,value,unit,source,population,geography
  - epidemiology_monthly.csv  : date,indicator,value,unit,source,population,geography
  - epidemiology_state.csv    : state,indicator,value,unit,source
  - variable_dictionary.csv   : indicator metadata (source, url, time range, unit, break)

Design notes:
  - Raw files are hand-curated from official sources (browser-verified), stored in
    module4_results/data/external/*.csv. This script only re-formats, never fabricates.
  - NSDUH methodological break at 2021 is flagged in variable_dictionary.
"""

import os
import csv
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = "/path/to/project"
# Must match the path downstream analyses read (s42/s45/s45c/s46):
#   PROJECT_ROOT/module4_results/data/external/processed/epidemiology_annual.csv
DATA_DIR = os.path.join(PROJECT_ROOT, "module4_results", "data", "external")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_csv(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required raw external file missing: {p}. Expected under {DATA_DIR} "
            "(see README.md / NSDUH_MDE_SOURCES.md). Refusing to continue with an "
            "empty table."
        )
    with open(p) as f:
        return list(csv.DictReader(f))


def build_annual():
    """Build unified annual epidemiology table."""
    rows = []

    # 1. CDC suicide rate (annual, 1990-2024)
    for r in load_csv("cdc_suicide_rate_1990_2024.csv"):
        rows.append({
            "year": int(r["year"]),
            "indicator": "suicide_rate",
            "value": float(r["suicide_rate_per_100k"]),
            "unit": "per_100k",
            "source": "CDC",
            "population": "US_all",
            "geography": "US",
        })

    # 2. NSDUH MDE prevalence (annual, 2015-2024, methodological break 2021)
    for r in load_csv("nsduh_mde_prevalence_2015_2024.csv"):
        rows.append({
            "year": int(r["year"]),
            "indicator": "MDE_prevalence",
            "value": float(r["mde_pct"]),
            "unit": "percent",
            "source": "NSDUH",
            "population": "US_18plus",
            "geography": "US",
        })

    rows.sort(key=lambda r: (r["indicator"], r["year"]))
    path = os.path.join(PROCESSED_DIR, "epidemiology_annual.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "indicator", "value", "unit", "source", "population", "geography"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved {path} ({len(rows)} rows)")


def build_monthly():
    """Build unified monthly epidemiology table."""
    rows = []
    # CDC monthly suicide (2020-2023, provisional)
    for r in load_csv("cdc_monthly_suicide_2020_2023.csv"):
        y, m = int(r["year"]), int(r["month"])
        rows.append({
            "date": f"{y:04d}-{m:02d}",
            "indicator": "suicide_deaths",
            "value": int(r["suicide_deaths_provisional"]),
            "unit": "count",
            "source": "CDC",
            "population": "US_all",
            "geography": "US",
        })
    rows.sort(key=lambda r: r["date"])
    path = os.path.join(PROCESSED_DIR, "epidemiology_monthly.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "indicator", "value", "unit", "source", "population", "geography"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved {path} ({len(rows)} rows)")


def build_state():
    """Build unified state-level table (for RQ4b spatial validation)."""
    rows = []
    for r in load_csv("mha_state_depression_2025.csv"):
        rows.append({
            "state": r["state"],
            "indicator": "adult_depression_prevalence",
            "value": float(r["adult_depression_pct"]),
            "unit": "percent",
            "source": "MHA",
            "year": 2025,
        })
    path = os.path.join(PROCESSED_DIR, "epidemiology_state.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "indicator", "value", "unit", "source", "year"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved {path} ({len(rows)} rows)")


def build_dictionary():
    """Variable dictionary for provenance.

    PROVENANCE (2026-09-07 audit): dataset IDs below still require verification
    against the live CDC/NSDUH pages before submission. Raw curated files live in
    module4_results/data/external/ (see README.md / NSDUH_MDE_SOURCES.md), not in
    the project source tree.
    """
    dict_rows = [
        {"indicator": "suicide_rate", "source": "CDC", "url": "data.cdc.gov/d/9j2v-jamp + d/w26f-tf3h",
         "time_start": 1990, "time_end": 2024, "frequency": "annual", "unit": "per_100k",
         "population": "US_all", "methodological_break": ""},
        {"indicator": "MDE_prevalence", "source": "NSDUH", "url": "SAMHSA Detailed Tables",
         "time_start": 2015, "time_end": 2024, "frequency": "annual", "unit": "percent",
         "population": "US_18plus", "methodological_break": "2021"},
        {"indicator": "suicide_deaths", "source": "CDC", "url": "data.cdc.gov/d/9dzk-mvmi",
         "time_start": 2020, "time_end": 2023, "frequency": "monthly", "unit": "count",
         "population": "US_all", "methodological_break": "provisional"},
        {"indicator": "adult_depression_prevalence", "source": "MHA", "url": "mhanational.org",
         "time_start": 2025, "time_end": 2025, "frequency": "cross-sectional", "unit": "percent",
         "population": "US_adults", "methodological_break": ""},
    ]
    path = os.path.join(PROCESSED_DIR, "variable_dictionary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["indicator", "source", "url", "time_start", "time_end", "frequency", "unit", "population", "methodological_break"])
        w.writeheader()
        w.writerows(dict_rows)
    logger.info(f"Saved {path}")


def main():
    logger.info("=== M4 external data pipeline ===")
    build_annual()
    build_monthly()
    build_state()
    build_dictionary()
    logger.info("DONE")


if __name__ == "__main__":
    main()
