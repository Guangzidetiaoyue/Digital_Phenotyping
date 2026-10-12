# -*- coding: utf-8 -*-
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("digital_phenotyping_reference_candidates.json", encoding="utf-8"))
for i, r in enumerate(d):
    print(f"{i:>3} [{r['score']:>2}] {r['journal'][:24]:24} {str(r['date'])[:10]:10} {r['doi'][:60]:60} | {r['title'][:110]}")