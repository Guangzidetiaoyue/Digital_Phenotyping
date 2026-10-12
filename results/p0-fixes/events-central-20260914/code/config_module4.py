"""
Configuration for Module 4: Macro digital epidemiology (calendar / events / external validation).

Legacy pipeline figures (s46_figures.py) use module4_results/figures/.
Publication figures (s46_figures_lyc.py) use figures_lyc/ with main/ and ed/ subdirs.
"""

import os

PROJECT_ROOT = "/path/to/project"

M4_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "module4_results")
DATA_DIR = os.path.join(M4_OUTPUT_DIR, "data")
STATS_DIR = os.path.join(M4_OUTPUT_DIR, "statistics")
M4_FIGURES_DIR = os.path.join(M4_OUTPUT_DIR, "figures")
FIGURES_DIR = os.path.join(M4_OUTPUT_DIR, "figures_lyc")

for _d in [M4_OUTPUT_DIR, DATA_DIR, STATS_DIR, M4_FIGURES_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

DISORDER_DISPLAY_NAMES = {
    "adhd": "ADHD",
    "anxiety": "Anxiety",
    "autism": "Autism",
    "bipolar": "Bipolar",
    "bpd": "BPD",
    "depression": "Depression",
    "eating_disorders": "Eating D.",
    "ocd": "OCD",
    "ptsd": "PTSD",
    "schizophrenia": "Schizophrenia",
}

# ---------------------------------------------------------------------------
# EVENT ANCHORS — single source of truth (anchors are MONTH resolution).
#   anchor    : event month used by the event-study/CAR analyses
#   label     : display label
#   est_start : start of the window-level DID estimation window (analyst-defined)
#   ev_end    : end of the window-level DID event window (analyst-defined)
#   source    : factual basis for the date (cite in Methods)
# ---------------------------------------------------------------------------
EVENTS = {
    "E1_election_2016": {
        "anchor": "2016-11-01", "label": "US Presidential Election 2016",
        "est_start": "2015-11-01", "ev_end": "2017-02-01",
        "source": "US presidential election day 8 Nov 2016"},
    "E2_celebrity_2018": {
        "anchor": "2018-06-01", "label": "Celebrity suicides 2018",
        "est_start": "2017-06-01", "ev_end": "2018-09-01",
        "source": "Kate Spade 5 Jun 2018; Anthony Bourdain 8 Jun 2018"},
    "E3_covid_2020": {
        "anchor": "2020-03-01", "label": "COVID-19 onset",
        "est_start": "2019-03-01", "ev_end": "2020-06-01",
        "source": ("WHO COVID-19 pandemic declaration 11 Mar 2020 "
                   "(https://www.who.int/director-general/speeches/detail/"
                   "who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020)")},
    "E4_blm_2020": {
        "anchor": "2020-05-01", "label": "George Floyd / BLM",
        "est_start": "2019-05-01", "ev_end": "2020-08-01",
        "source": "killing of George Floyd 25 May 2020"},
    "E5_capitol_2021": {
        "anchor": "2021-01-01", "label": "US Capitol attack",
        "est_start": "2020-01-01", "ev_end": "2021-04-01",
        "source": "US Capitol attack 6 Jan 2021"},
}

# Event-study windows (analyst-defined).
EST_WINDOW = (-12, -2)   # estimation window (months relative to event)
EV_WINDOW = (0, 3)       # primary event window
EVENT_WINDOWS = [(0, 3), (0, 1), (0, 6)]  # main + sensitivity

# COVID-19 breakpoint (WHO pandemic declaration) and analyst-defined phases.
COVID_BREAKPOINT = "2020-03"
COVID_BREAKPOINT_SOURCE = ("WHO COVID-19 pandemic declaration 11 Mar 2020 "
                           "(https://www.who.int/director-general/speeches/detail/"
                           "who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020)")
# acute = first wave/lockdowns; adaptation = post-first-wave; vaccine = post-US
# rollout; new_normal = 2022 onward. Analyst-defined; cite rationale in Methods.
COVID_PHASES = [
    ("acute", "2020-03", "2020-06"),
    ("adaptation", "2020-07", "2021-03"),
    ("vaccine", "2021-04", "2022-01"),
    ("new_normal", "2022-02", "2022-12"),
]
