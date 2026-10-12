# -*- coding: utf-8 -*-
"""Verify pre-2023 social-media mental-health classics via Semantic Scholar API."""
import json, sys, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")

QUERIES = [
    "Predicting depression via social media De Choudhury",
    "Quantifying mental health signals in Twitter Coppersmith",
    "Digital phenotyping technologies for mental health",
    "The online social self open vocabulary Kern",
]

def q(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(query) + "&limit=4&fields=title,year,venue,externalIds,abstract,authors"
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("data", [])

for query in QUERIES:
    print("#" * 30, query)
    try:
        for it in q(query):
            a = ", ".join(x.get("name","") for x in (it.get("authors") or [])[:4])
            ei = it.get("externalIds") or {}
            print(f"  {it.get('year')} | {ei.get('DOI')} | {ei.get('ArXiv')} | {it.get('venue','')[:30]:30} | {it.get('title','')[:90]}\n       [{a[:80]}]")
    except Exception as e:
        print("  ERR", e)
    print()