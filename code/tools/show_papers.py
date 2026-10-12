# -*- coding: utf-8 -*-
"""Show full details (journal/date/doi/abstract) for manually selected DOIs."""
import json, sys, argparse
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--dois", nargs="*", default=[])
parser.add_argument("--search", nargs="*", default=[])
parser.add_argument("--kw", nargs="*", default=[])
args = parser.parse_args()

with open("raw_papers_2023_2026.json", encoding="utf-8") as f:
    data = json.load(f)

JOURNALS = [k for k in data.keys() if k != "fetch_info"]
def norm(d):
    d = (d or "").lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")
    return d
idx = {}
for j in JOURNALS:
    for p in data[j]:
        d = norm(p.get("doi"))
        if d:
            idx[d] = p

def show(p):
    print("=" * 100)
    print("TITLE:", p.get("title"))
    print("JOURNAL:", p.get("journal"), "| DATE:", p.get("published_date"))
    print("DOI:", p.get("doi"))
    print("AUTHORS:", p.get("authors"))
    a = p.get("abstract")
    if not a or a.strip().lower() in ("no abstract available.", "no abstract available"):
        a = "(NO ABSTRACT AVAILABLE)"
    print("ABSTRACT:", a[:1800])
    print()

if args.dois:
    for d in args.dois:
        key = norm(d)
        if key in idx:
            show({**idx[key], "journal": "LOOKUP"})
        else:
            print("NOT FOUND:", d)

if args.search or args.kw:
    for j in JOURNALS:
        for p in data[j]:
            t = (p.get("title") or "").lower()
            a = (p.get("abstract") or "").lower()
            text = t + " " + a
            hit = all(kw in text for kw in args.kw) if args.kw else False
            if args.search:
                hit = hit or any(s in text for s in args.search)
            if hit:
                show({**p, "journal": j})