# -*- coding: utf-8 -*-
"""Look up exact DOI for a list of titles in the JSON."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open("raw_papers_2023_2026.json", encoding="utf-8") as f:
    data = json.load(f)
JOURNALS = [k for k in data.keys() if k != "fetch_info"]

TITLES = [
    "Simulating human well-being with large language models",
    "Social media interventions to improve well-being",
    "Social media use in adolescents with and without mental health conditions",
    "Differences in social media use between adolescents with and without mental health conditions",
    "Time spent on social media among the least influential factors",
    "Political protests, social media use and mental well-being",
    "Interpersonal conflicts, social media use and depression associated with protests",
    "The promise of digital phenotyping for the early detection",
    "The continued hype and hope of digital phenotyping",
    "Key language markers of depression on social media depend on race",
    "Reply to Wang: Clarifying model performance",
    "Do \u201cBlack individuals\u201d really display no linguistic markers of depression",
    "Reply to De Freitas",
    "Language sentiment predicts changes in depressive symptoms",
    "Do smartphones and social media really harm teens",
    "Transdiagnostic network for psychiatric illness derived from atrophy and lesions",
    "Diversity and the Hierarchical Taxonomy",
    "\u201cP factor\u201d could open an important window",
    "Depression and anxiety characterization and detection with multimodal deep learning",
    "Early detection of autism using digital behavioral phenotyping",
    "Digital phenotyping could help detect autism",
]

for t in TITLES:
    hits = []
    for j in JOURNALS:
        for p in data[j]:
            tt = (p.get("title") or "").lower()
            if t.lower() in tt:
                hits.append((j, p.get("doi"), p.get("title"), p.get("published_date"), p.get("authors")))
    if hits:
        for j, d, tt, dt, au in hits:
            print(f"TITLE[{t[:40]:40}] -> {j[:18]:18} | {d} | {dt} | {tt[:80]}")
    else:
        print(f"TITLE[{t[:40]:40}] -> NOT FOUND")