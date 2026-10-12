# Robust but scale-dependent digital phenotypes of mental disorders

Code and derived results accompanying the manuscript
*"Robust but scale-dependent digital phenotypes of mental disorders in 128,000 Reddit users"*.

This repository is provided for **double-blind peer review**. It contains no author-identifying
information; filesystem paths in the code have been replaced with the placeholders
`/path/to/project` and `/path/to/code`, and host names with `remote-host`.

## What this repository contains

| Path | Contents |
|---|---|
| `code/` | Analysis pipeline: cohort construction (`phase1_finder.py`, `phase2_extractor.py`, `post_processing.py`), feature engineering, discrimination, cross-diagnostic structure, trajectories, population/audit analyses, and the figure-generation scripts (`code/figures_NMH/`). |
| `data/` | Externally sourced epidemiological series used for validation (CDC suicide mortality; SAMHSA NSDUH major-depressive-episode prevalence; state MHA depression). |
| `results/` | Derived result tables (aggregate statistics) underlying every figure and claim. |
| `manuscript/` | LaTeX source of the main text, Extended Data and Supplementary Information, and the final figure PDFs. |

## Analysis outline

The study assembles a matched cohort of Reddit users who self-reported a diagnosis and tests
digital phenotypes at five scales of inference (individual detection; disorder-versus-disorder
separation; latent structure; individual early warning; population monitoring), plus a
composition-induced temporal bias (CITB) audit. Key entry points:

- Cohort/PSM: `code/post_processing.py`, `code/run_pipeline0104_0302.py`
- Features: `code/Module1/`
- Trajectories / temporal boundary: `code/Module2/`
- Cross-diagnostic structure: `code/Module3/`
- Population monitoring and CITB audit: `code/Module4/`
- Figures: `code/figures_NMH/`

## Running the code

The pipeline reads a project root that is not redistributed (raw, user-level Reddit data are
withheld to protect privacy). To run against your own data layout, set the base path at the top
of the relevant scripts, replacing the placeholder:

```python
PROJECT_ROOT = "/path/to/project"   # <- set to your data root
CODE_ROOT    = "/path/to/code"
```

Required packages: `python 3.11`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`,
`ruptures`, `spacy`, `vaderSentiment`, `pyarrow`. See `code/Module*/config_module*.py` for
analysis parameters (window sizes, penalties, seeds).

## Data availability

Publicly posted Reddit content was analysed in anonymised form; raw user-level data are **not**
redistributed to protect privacy. Cohort summary statistics and the derived result tables
underlying every figure (`results/`) and the externally sourced epidemiological series
(`data/`) are provided here.

## Ethics

All data are public Reddit posts analysed in anonymised form; the self-reported diagnosis label
is a behavioural proxy and not a clinical diagnosis. No contact, intervention or
re-identification was performed.

## License

Code released under the MIT License (see `LICENSE`); derived data under CC-BY-4.0.
