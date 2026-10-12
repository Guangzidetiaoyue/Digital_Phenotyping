"""
Module 3 - Analysis B1: Multi-Diagnosis Parser
=============================================
Extract ALL self-reported diagnoses per user from silver diagnosis posts,
enabling comorbidity "intermediate-state" analyses (a user's digital phenotype
should lie between the centroids of their diagnosed disorders).

Method:
  B1.1 Scan all silver partitions for `is_diagnosis_post == True` posts.
  B1.2 For each post, find candidate disorders via MENTAL_DISORDERS keywords.
  B1.3 Validate each candidate with spaCy syntactic checks (first-person
       diagnosis: "I was diagnosed with X" / "I have X") + exclusion keywords.
  B1.4 Aggregate per-user disorder SET (may contain >1 disorder).

Output:
  - user_multi_disorder.parquet: user_id_hash -> disorder set (encoded)
  - multi_disorder_stats.json: counts of comorbidity patterns
"""

import os
import sys
import json
import glob
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/path/to/code")
import spacy
from config import MENTAL_DISORDERS, EXCLUSION_KEYWORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SILVER_DATA_DIR = "/path/to/project/cohort_v1/silver_v2/data"
OUT_DIR = "/path/to/project/module3_results/statistics"

# Build term -> disorder map (longest-first for precise matching)
TERM_MAP = []
for disorder, terms in MENTAL_DISORDERS.items():
    for t in terms:
        TERM_MAP.append((t.lower(), disorder))
TERM_MAP.sort(key=lambda x: len(x[0]), reverse=True)

# Negation / non-first-person cues
NEGATION_CUES = [
    "don't have", "do not have", "never had", "not diagnosed", "never diagnosed",
    "wasn't diagnosed", "was not diagnosed", "aren't diagnosed", "not depressed",
    "not anxious", "don't think", "don't believe", "not sure if", "probably",
    "might have", "suspect", "not officially", "self-diagnos", "undiagnosed",
    "misdiagnos", "my mom", "my dad", "my friend", "my partner", "my wife",
    "my husband", "my brother", "my sister", "my family", "my son", "my daughter",
]


def parse_candidates(text):
    """Find candidate disorders mentioned in text (pre-validation)."""
    tl = text.lower()
    found = set()
    for term, disorder in TERM_MAP:
        if term in tl:
            found.add(disorder)
    return found


def validate_first_person(text, disorder, nlp):
    """Check the disorder is a first-person diagnosis (not denial/third-party).

    Fast path: regex negation/exclusion + 'I have/diagnosed with' patterns.
    SpaCy only used when regex is inconclusive.
    """
    import re
    tl = text.lower()
    # 1. Exclusion cues (denial / third-party / uncertainty)
    if any(cue in tl for cue in NEGATION_CUES):
        return False
    if any(ex in tl for ex in EXCLUSION_KEYWORDS):
        return False

    # 2. Regex first-person diagnosis patterns (fast)
    for d_keyword in MENTAL_DISORDERS[disorder]:
        kw = re.escape(d_keyword)
        if re.search(rf"i\s+(?:was|have\s+been|am|'ve\s+been)\s+(?:diagnosed|diagnosis)\s+(?:with|of)\s+{kw}", tl):
            return True
        if re.search(rf"i\s+have\s+{kw}", tl):
            return True
        if re.search(rf"i\s+have\s+been\s+(?:living|dealing|suffering)\s+with\s+{kw}", tl):
            return True

    # 3. SpaCy confirm only for 'diagnose' verb sentences (rare, slow)
    try:
        # only parse if text contains a 'diagnos' verb
        if "diagnos" not in tl:
            return False
        doc = nlp(text[:1500])
        for token in doc:
            if token.lemma_ == "diagnose":
                has_first_person = any(
                    c.text.lower() in ("i", "me") and c.dep_ in ("nsubj", "nsubjpass")
                    for c in token.children
                )
                negated = any(c.dep_ == "neg" for c in token.children)
                if has_first_person and not negated:
                    return True
    except Exception:
        pass
    return False


def scan_all_partitions(nlp, max_partitions=None):
    """Scan all (or sampled) partitions, return per-user disorder sets."""
    parts = sorted(glob.glob(os.path.join(SILVER_DATA_DIR, "part_*")))
    if max_partitions:
        parts = parts[:max_partitions]
    logger.info(f"Scanning {len(parts)} partitions...")

    user_disorders = defaultdict(set)
    n_posts = 0
    n_multi_posts = 0
    # Precompute regex for candidate terms (single pass per text)
    import re
    term_regexes = [(re.compile(re.escape(t)), d) for t, d in TERM_MAP]

    for pi, p in enumerate(parts):
        for f in glob.glob(os.path.join(p, "*.parquet")):
            try:
                # Read only diagnosis posts via pyarrow filter (avoids loading
                # all rows; diagnosis posts are sparse ~0.1% of rows)
                table = pq.read_table(
                    f,
                    columns=["user_id_hash", "clean_text", "is_diagnosis_post"],
                    filters=[("is_diagnosis_post", "==", True)],
                )
                df = table.to_pandas()
            except Exception:
                # Fallback: read all rows, filter in pandas
                try:
                    df_full = pq.read_table(
                        f, columns=["user_id_hash", "clean_text", "is_diagnosis_post"]
                    ).to_pandas()
                    df = df_full[df_full["is_diagnosis_post"] == True]
                except Exception:
                    continue
            for uid, text in zip(df["user_id_hash"], df["clean_text"]):
                n_posts += 1
                if not isinstance(text, str) or len(text.strip()) < 10:
                    continue
                tl = text.lower()
                candidates = set()
                for rx, disorder in term_regexes:
                    if rx.search(tl):
                        candidates.add(disorder)
                if not candidates:
                    continue
                validated = set()
                for d in candidates:
                    if validate_first_person(text, d, nlp):
                        validated.add(d)
                if len(validated) > 1:
                    n_multi_posts += 1
                user_disorders[uid].update(validated)
        if (pi + 1) % 50 == 0:
            logger.info(f"  scanned {pi+1}/{len(parts)} partitions, "
                        f"{len(user_disorders)} users")

    logger.info(f"Done: {n_posts} diag posts, {len(user_disorders)} users with diagnoses")
    return user_disorders, n_posts, n_multi_posts


def main():
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy loaded.")

    user_disorders, n_posts, n_multi_posts = scan_all_partitions(nlp)

    # Encode disorder sets
    disorders = sorted(MENTAL_DISORDERS.keys())
    rows = []
    for uid, dset in user_disorders.items():
        rows.append({
            "user_id_hash": uid,
            "n_disorders": len(dset),
            **{f"has_{d}": int(d in dset) for d in disorders},
        })
    udf = pd.DataFrame(rows)
    udf.to_parquet(os.path.join(OUT_DIR, "user_multi_disorder.parquet"), index=False)
    logger.info(f"Saved user_multi_disorder.parquet: {len(udf)} users")

    # Statistics
    n_multi_users = (udf["n_disorders"] > 1).sum()
    dist = udf["n_disorders"].value_counts().sort_index()
    logger.info(f"\n=== Multi-diagnosis statistics ===")
    logger.info(f"  total users with diagnosis: {len(udf)}")
    logger.info(f"  users with >1 disorder: {n_multi_users} ({n_multi_users/len(udf)*100:.1f}%)")
    logger.info(f"  n_disorders distribution: {dist.to_dict()}")
    logger.info(f"  multi-disorder posts: {n_multi_posts}/{n_posts} ({n_multi_posts/n_posts*100:.1f}%)")

    # Top comorbidity pairs (among users with >=2)
    pair_counter = Counter()
    for _, row in udf[udf["n_disorders"] >= 2].iterrows():
        has = [d for d in disorders if row[f"has_{d}"]]
        for i in range(len(has)):
            for j in range(i + 1, len(has)):
                pair_counter[tuple(sorted([has[i], has[j]]))] += 1
    top_pairs = pair_counter.most_common(20)
    logger.info(f"\n  Top comorbidity pairs:")
    for (a, b), c in top_pairs:
        logger.info(f"    {a}-{b}: {c}")

    stats = {
        "n_diag_posts": n_posts,
        "n_multi_diag_posts": n_multi_posts,
        "n_users_with_diagnosis": len(udf),
        "n_users_multi": int(n_multi_users),
        "pct_users_multi": float(n_multi_users / len(udf)) if len(udf) else 0,
        "n_disorders_dist": {str(k): int(v) for k, v in dist.items()},
        "top_comorbidity_pairs": [{"pair": list(k), "n": v} for k, v in top_pairs],
    }
    with open(os.path.join(OUT_DIR, "multi_diagnosis_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("DONE")


if __name__ == "__main__":
    main()
