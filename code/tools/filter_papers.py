# -*- coding: utf-8 -*-
"""Filter raw_papers_2023_2026.json for papers closely related to the Reddit
longitudinal digital phenotyping study. Writes candidates to a JSON for curation."""
import json, re, sys

SRC = "raw_papers_2023_2026.json"
OUT = "digital_phenotyping_reference_candidates.json"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

JOURNALS = [k for k in data.keys() if k != "fetch_info"]
print("Journals:", JOURNALS)
n_total = sum(len(data[j]) for j in JOURNALS)
print("Total articles:", n_total)

def has(paper, *fields):
    s = ""
    for f in fields:
        v = paper.get(f, "")
        if v:
            s += " " + v
    return s.lower()

def kw(s, *words):
    return any(w in s for w in words)

# Category rules: each returns score contribution
RULES = {
    "digital_phenotyping": lambda t, a: kw(t, "digital phenotyp", "digital biomarker") * 4 + kw(a, "digital phenotyp") * 2,
    "social_media_mental": lambda t, a: (kw(t, "reddit") or kw(t, "social media", "twitter", "facebook", "tiktok", "online forum") ) and (kw(t, "depress", "suicid", "anxiety", "psychiatr", "schizophrenia", "bipolar", "mental health", "ptsd", "eating disorder", "self-harm", "self harm")),
    "language_mental": lambda t, a: (kw(t, "language", "linguistic", "nlp", "natural language", "text analysis", "liwc", "verbal")) and (kw(t, "depress", "suicid", "psychiatr", "psychosis", "schizophrenia", "anxiety", "mental health", "bipolar", "ptsd")),
    "llm_mental": lambda t, a: (kw(t, "large language model", "llm", "chatgpt", "gpt-", "chatbot")) and (kw(t, "mental health", "depress", "psychiatr", "suicid", "anxiety", "therapy", "counsel", "therapeutic")),
    "trajectory_digital": lambda t, a: (kw(t, "trajector")) and (kw(t, "depress", "mood", "mental health", "anxiety", "psychiatr", "suicid", "well-being", "digital", "social media", "language")),
    "passive_sensing": lambda t, a: (kw(t, "passive sensing", "smartphone", "digital phenotyp", "wearable", "ambulatory")) and (kw(t, "mental health", "depress", "mood", "anxiety", "psychiatr", "suicid", "well-being")),
    "suicide_prediction": lambda t, a: kw(t, "suicide", "suicidal", "self-harm", "self harm") and (kw(t, "predict", "risk", "detect", "social media", "language", "machine learning", "deep learning", "marker") or kw(a, "social media")),
    "transdiagnostic": lambda t, a: (kw(t, "transdiagnostic", "p factor", "general psychopathology", "hierarchical taxonomy", "hitop")),
    "causal_socialmedia": lambda t, a: (kw(t, "difference-in-difference", "difference in difference", "event study", "natural experiment", "quasi-experimental")) and (kw(t, "social media", "digital", "online")) and (kw(t, "mental health", "well-being", "wellbeing", "depress", "anxiety")),
    "ml_psychiatric": lambda t, a: (kw(t, "machine learning", "deep learning", "neural network", "ai")) and (kw(t, "depression", "depressive", "suicide", "schizophrenia", "psychosis", "psychiatric", "mental disorder", "mental health") and kw(t, "predict", "detect", "classif")),
    "mh_subreddit": lambda t, a: kw(t, "subreddit"),
    "psm_case_control": lambda t, a: (kw(t, "propensity", "case-control", "case control", "matched cohort")) and (kw(t, "social media", "digital", "mental", "depress", "suicid")),
    "efficacy_treatment_digital": lambda t, a: (kw(t, "digital intervention", "app-based", "digital therapeutic", "internet-based")) and (kw(t, "mental health", "depress", "anxiety")),
}

results = []
for j in JOURNALS:
    for p in data[j]:
        title = (p.get("title") or "").lower()
        abst = (p.get("abstract") or "").lower()
        if abst in ("no abstract available.", "", "no abstract available"):
            abst = ""
        score = 0
        cats = []
        for name, rule in RULES.items():
            s = 0
            try:
                s = rule(title, abst)
            except TypeError:
                s = rule(title, abst)
            if isinstance(s, bool):
                s = int(s) * 2
            if s:
                if isinstance(s, bool):
                    s = int(s)
                score += max(0, s)
                if s >= 2:
                    cats.append(name)
        # title-level direct hits: boost
        if kw(title, "digital phenotyp", "social media", "reddit", "natural language") and kw(title, "depress", "suicid", "mental", "psychiatr", "anxiety", "mood", "well-being"):
            score += 3
            if "title_hit" not in cats:
                cats.append("title_hit")
        if score >= 2 and cats:
            results.append({
                "journal": j,
                "title": p.get("title"),
                "authors": p.get("authors"),
                "doi": p.get("doi"),
                "date": p.get("published_date"),
                "cats": cats,
                "score": score,
                "abstract": p.get("abstract") or ""
            })

# dedupe by DOI
seen = {}
for r in results:
    d = (r["doi"] or "").lower().strip("/")
    if d and d in seen:
        continue
    seen[d] = r

results = list(seen.values())
print(f"Candidates: {len(results)}")
from collections import Counter
cc = Counter()
for r in results:
    for c in r["cats"]:
        cc[c] += 1
for c, n in cc.most_common():
    print(f"  {c}: {n}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(f"Saved -> {OUT}")
