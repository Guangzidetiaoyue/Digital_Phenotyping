# -*- coding: utf-8 -*-
"""Verify classic pre-2023 papers via CrossRef API (real DOIs / titles / journals)."""
import json, sys, urllib.request, urllib.parse

sys.stdout.reconfigure(encoding="utf-8")

CHECKS = [
    ("10.1073/pnas.1802331115", "Eichstaedt 2018 Facebook depression"),     # well-known DOI, verify
    ("10.1177/2167702613497473", "Caspi 2014 p factor"),
    ("10.1037/abn0000258", "Kotov 2017 HiTOP"),
    ("10.1371/journal.pone.0073791", "Schwartz 2013 open vocabulary"),
    ("10.1002/wps.20490", "Insel 2017 digital phenotyping"),
    ("10.1038/tp.2017.25", "Torous 2017 RDoC"),
    ("10.1038/npp.2016.26", "Onnela 2016 smartphone phenotyping"),
    ("10.1016/j.cobeha.2017.07.005", "Guntuku 2017 review"),
    ("10.1080/02699940441000030", "Rude 2004 depression language"),
    ("10.1177/2167702617747074", "Al-Mosaiwi 2018 absolutist"),
    ("10.1140/epjds/s13688-017-0110-1", "Reece 2017 Instagram depression"),
    ("10.1038/s41562-018-0506-1", "Orben 2019 adolescent well-being digital"),
    ("10.1257/aer.20211218", "Braghieri 2022 social media mental health AER"),
    ("10.1037/0022-3514.77.6.1296", "Pennebaker 1999 linguistic styles"),
    ("10.1097/00006842-200109000-00010", "Stirman 2001 poetry suicidal"),
    ("10.1038/s41562-019-0714-3", "Przybylski 2019 pubertal sensitivity"),
]

def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]

for doi, label in CHECKS:
    try:
        m = fetch(doi)
        title = (m.get("title") or [""])[0]
        year = m.get("issued", {}).get("date-parts", [[None]])[0][0]
        cont = (m.get("container-title") or [""])[0] if isinstance(m.get("container-title"), list) else ""
        print(f"OK   {doi} | {year} | {cont[:40]:40} | {label}\n     -> {title[:110]}")
    except Exception as e:
        print(f"FAIL {doi} | {label} | {e}")