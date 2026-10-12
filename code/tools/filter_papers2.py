# -*- coding: utf-8 -*-
"""Second-stage targeted filter for digital phenotyping / social media mental health."""
import json, re, sys
sys.stdout.reconfigure(encoding="utf-8")

SRC = "raw_papers_2023_2026.json"
OUT = "digital_phenotyping_reference_candidates2.json"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

JOURNALS = [k for k in data.keys() if k != "fetch_info"]

def fields(p):
    t = (p.get("title") or "").lower()
    a = (p.get("abstract") or "").lower()
    na = a in ("no abstract available.", "no abstract available", "", "n/a")
    return t, a, na

# Strict rules: manual curation; each returns a category label + reason or None
def classify(t, a, na):
    def has(s, *ws):
        return any(w in s for w in ws)
    # Digital phenotyping explicit
    if has(t, "digital phenotyp"):
        return "digital_phenotyping", "title:digital phenotyp"
    if has(a, "digital phenotyp") and (has(t, "depress", "mental", "psych", "suicid", "mood", "anxiety") or has(a, "depress", "mental health")):
        return "digital_phenotyping", "abstract:digital phenotyp"
    # Reddit specifically
    if has(t, "reddit"):
        if has(t, "mental", "depress", "suicid", "psych", "anxiety", "disorder", "well"):
            return "reddit_mental", "title:reddit"
        return "reddit_other", "title:reddit"
    if has(a, "reddit") and has(t, "mental", "depress", "suicid", "psych", "anxiety"):
        return "reddit_mental", "abstract:reddit"
    # Social media x mental condition (title strong)
    if has(t, "social media", "facebook", "instagram", "tiktok", "twitter", "x.com") and has(t, "mental health", "depression", "depressive", "suicidal", "suicide", "anxiety", "psychiatric", "well-being", "wellbeing", "self-harm", "self harm"):
        return "socialmedia_mental", "title:sm+mental"
    # Language markers of depression / suicide
    if (has(t, "language", "linguistic", "liwc", "text analysis", "natural language", "word use", "linguistic marker", "language marker", "sentiment")) and (has(t, "depress", "suicidal", "suicide", "mental health", "psychiatric", "psychosis", "bipolar", "mood disorder", "schizophrenia")):
        return "language_mental", "title:lang+mental"
    # LLM mental health prediction / detection
    if (has(t, "large language model", "llm", "chatgpt", "gpt")) and (has(t, "depress", "mental health", "psychiatr", "suicid", "anxiety", "counsel", "therap", "well-being", "wellbeing")):
        return "llm_mental", "title:llm+mental"
    # Smartphone / passive sensing digital
    if (has(t, "smartphone", "digital phenotyp", "passive sens", "wearable", "digital marker", "digital biomarker", "ambulatory")) and (has(t, "depress", "mental health", "mental disorder", "psychiatr", "suicid", "mood", "anxiety", "well-being")):
        return "smartphone_mental", "title:smart+mental"
    # p-factor / g-factor transdiagnostic structure
    if has(t, "p factor", "general factor of psychopathology", "g factor", "hierarchical taxonomy", "hitop", "transdiagnostic structure", "general psychopathology"):
        if has(t, "language", "digital", "social media", "behavioral", "behavior"):
            return "p_factor_lang", "title:p-factor+digital"
        return "p_factor", "title:p-factor"
    # Trajectories of mental conditions (longitudinal)
    if has(t, "trajector") and has(t, "depress", "anxiety", "mental health", "psychiatr", "mood", "suicid", "bipolar"):
        if has(t, "social media", "digital", "language", "online"):
            return "trajectory_digital", "title:traj+digital"
        return "trajectory_mental", "title:traj+mental"
    # Causal / natural experiment social media mental
    if has(t, "difference-in-difference", "natural experiment", "quasi-experiment", "event study", "causal") and has(t, "social media", "facebook", "smartphone") and has(t, "mental", "depression", "well-being", "wellbeing", "anxiety", "self-harm"):
        return "causal_sm", "title:causal+sm"
    return None, None

matched = []
for j in JOURNALS:
    for p in data[j]:
        t, a, na = fields(p)
        cat, reason = classify(t, a, na)
        if cat:
            matched.append({
                "journal": j, "title": p.get("title"), "authors": p.get("authors"),
                "doi": p.get("doi"), "date": p.get("published_date"),
                "cat": cat, "reason": reason,
                "has_abstract": not na, "abstract": p.get("abstract") or ""
            })

# dedupe by DOI
seen = {}
for r in matched:
    d = (r["doi"] or "").lower().strip("/")
    if d and d in seen:
        # keep the one with more info
        continue
    seen[d] = r
matched = list(seen.values())

from collections import Counter
cc = Counter(r["cat"] for r in matched)
print("Total:", len(matched))
for c, n in cc.most_common():
    print(f"  {c}: {n}")
for r in matched:
    ab = "(no abs)" if not r["has_abstract"] else ""
    print(f"[{r['cat']}] [{r['date'][:7]}] {r['journal'][:24]:24} {ab:8} {r['title'][:105]}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(matched, f, ensure_ascii=False, indent=1)