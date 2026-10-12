"""
Configuration for Module 3: Cross-disorder distinguishability, geometry, networks, comorbidity.

Legacy pipeline scripts (s31–s35) may write previews under module3_results/figures/.
Publication figures (s36_figures_lyc.py) use figures_lyc/ with main/ and ed/ subdirs.
"""

import os

PROJECT_ROOT = "/path/to/project"

M3_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "module3_results")
STATS_DIR = os.path.join(M3_OUTPUT_DIR, "statistics")
M3_FIGURES_DIR = os.path.join(M3_OUTPUT_DIR, "figures")
FIGURES_DIR = os.path.join(M3_OUTPUT_DIR, "figures_lyc")
TEMP_DIR = "/tmp/module3_temp"

for _d in [M3_OUTPUT_DIR, STATS_DIR, M3_FIGURES_DIR, FIGURES_DIR, TEMP_DIR]:
    os.makedirs(_d, exist_ok=True)

DISORDER_DISPLAY_NAMES = {
    "depression": "Depression",
    "bipolar": "Bipolar",
    "anxiety": "Anxiety",
    "ocd": "OCD",
    "ptsd": "PTSD",
    "bpd": "Borderline PD",
    "eating_disorders": "Eating Disorders",
    "schizophrenia": "Schizophrenia",
    "adhd": "ADHD",
    "autism": "Autism",
    "control": "Control",
}

HITOP_SPECTRA = {
    "Internalizing": ["depression", "anxiety", "eating_disorders"],
    "Thought Disorder": ["schizophrenia", "bipolar"],
    "Externalizing": ["adhd"],
    "Detachment/Personality": ["bpd", "ptsd", "ocd"],
    "Neurodevelopmental": ["autism"],
}

HITOP_COLORS = {
    "Internalizing": "#E74C3C",
    "Thought Disorder": "#9B59B6",
    "Externalizing": "#F39C12",
    "Detachment/Personality": "#3498DB",
    "Neurodevelopmental": "#2ECC71",
}

DSM5_CHAPTERS = {
    "Depressive/Bipolar": ["depression", "bipolar"],
    "Anxiety/OCD/Trauma": ["anxiety", "ocd", "ptsd"],
    "Schizophrenia": ["schizophrenia"],
    "Neurodevelopmental": ["adhd", "autism"],
    "Feeding/Eating": ["eating_disorders"],
    "Personality": ["bpd"],
}
# DSM-5-INFORMED grouping (merged chapters for our 10-disorder scheme);
# NOT a literal listing of DSM-5 chapters.
DSM5_INFORMED_CHAPTERS = DSM5_CHAPTERS

COMORBIDITY_CSV = os.path.join(STATS_DIR, "external_comorbidity.csv")


class ComorbidityMatrix:
    """Loaded external per-pair comorbidity matrix (single source of truth).

    Reads external_comorbidity.csv (columns disorder_a, disorder_b, value,
    measure, source, ref) from STATS_DIR. Raises FileNotFoundError if absent:
    no uncited fallback matrix is permitted. 'value' holds the measure given
    in 'measure' (currently adjusted/pooled odds ratios).
    """

    def __init__(self, path=None):
        import pandas as pd
        path = path or COMORBIDITY_CSV
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"External comorbidity matrix not found: {path}. "
                "Refusing to use an uncited fallback matrix."
            )
        self.path = path
        self.df = pd.read_csv(path)
        need = {"disorder_a", "disorder_b", "value"}
        missing = need - set(self.df.columns)
        if missing:
            raise ValueError(f"external_comorbidity.csv missing columns {missing}")
        self.pairs = {
            tuple(sorted([str(r["disorder_a"]), str(r["disorder_b"])])): float(r["value"])
            for _, r in self.df.iterrows()
        }
        self.measure = (str(self.df["measure"].dropna().iloc[0])
                        if "measure" in self.df.columns and self.df["measure"].notna().any()
                        else "unspecified")
        refs = ([str(x) for x in self.df["ref"].dropna().tolist()]
                if "ref" in self.df.columns else [])
        self.refs = sorted(set(refs))
        self.source_note = "; ".join(self.refs)[:800]

    def get(self, a, b):
        return self.pairs.get(tuple(sorted([str(a), str(b)])))

    def to_matrix(self, disorders):
        import numpy as np
        idx = {d: i for i, d in enumerate(disorders)}
        M = np.full((len(disorders), len(disorders)), np.nan)
        for (a, b), v in self.pairs.items():
            if a in idx and b in idx:
                M[idx[a], idx[b]] = v
                M[idx[b], idx[a]] = v
        return M


def load_comorbidity_matrix(disorders, path=None):
    """Return (matrix, ComorbidityMatrix) for the given disorder order."""
    cm = ComorbidityMatrix(path=path)
    return cm.to_matrix(disorders), cm


RANDOM_SEED = 42
