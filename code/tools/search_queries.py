# -*- coding: utf-8 -*-
"""Run several targeted queries; print compact title/DOI lines per query."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("raw_papers_2023_2026.json", encoding="utf-8") as f:
    data = json.load(f)
JOURNALS = [k for k in data.keys() if k != "fetch_info"]

QUERIES = [
    ("reddit", ["reddit"]),
    ("social media + depression", ["depression", "social media"]),
    ("language/depression", ["language", "depression"]),
    ("LLM + mental", ["large language model", "mental health"]),
    ("suicide + social/language", ["suicide", "language"]),
    ("passive sensing", ["passive sensing"]),
    ("digital trajectory/cohort", ["trajectories", "depression"]),
    ("eating disorder + digital", ["eating disorder"]),
    ("mood + smartphone", ["smartphone", "mood"]),
    ("NLP + psychiatric", ["natural language processing", "psychiatric"]),
]

for qname, kws in QUERIES:
    print("#" * 30, qname)
    seen = set()
    for j in JOURNALS:
        for p in data[j]:
            t = (p.get("title") or "").lower()
            a = (p.get("abstract") or "").lower()
            text = t + a
            if all(kw in text for kw in kws):
                d = (p.get("doi") or "").lower().strip("/")
                if d in seen:
                    continue
                seen.add(d)
                verb = "T:" if (t and all(kw in t for kw in kws)) else "  "
                print(f"  {verb} [{j[:14]}] {d[:52]:52} | {p.get('title','')[:95]}")
    print()