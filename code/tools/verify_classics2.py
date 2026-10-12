# -*- coding: utf-8 -*-
"""Search CrossRef by title to find correct DOIs for classics."""
import json, sys, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")

QUERIES = [
    "Digital phenotyping a global tool for psychiatry Insel World Psychiatry",
    "Harnessing Smartphone-Based Digital Phenotyping Onnela Rauch",
    "Language use of depressed and depression-vulnerable college students Rude",
    "Instagram photos reveal predictive markers of depression Reece Danforth",
    "Word use in the poetry of suicidal and nonsuicidal poets",
    "The welfare effects of social media Allcott",
]

def q(query):
    url = "https://api.crossref.org/works?rows=3&select=DOI,title,issued,container-title&query.bibliographic=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]

for query in QUERIES:
    print("#" * 30, query)
    try:
        for it in q(query):
            t = (it.get("title") or [""])[0]
            y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
            c = (it.get("container-title") or [""])[0]
            print(f"  {it['DOI']:45} {y} | {c[:35]:35} | {t[:95]}")
    except Exception as e:
        print("  ERR", e)
    print()