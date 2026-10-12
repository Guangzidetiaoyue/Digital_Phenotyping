#!/bin/bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
ROOT=/path/to/code
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_events.txt 2>&1 || true
cd "$ROOT/Module4"
python s44_event_study.py; echo "STATUS s44=$?"
python s44c_event_corrected.py; echo "STATUS s44c=$?"
python s44f_final_events.py; echo "STATUS s44f=$?"
python s43_covid_its.py; echo "STATUS s43=$?"
python s43c_covid_its_corrected.py; echo "STATUS s43c=$?"
python audit_event_validate.py; echo "STATUS audit_ev=$?"
python audit_windowlevel_validate.py; echo "STATUS audit_wl=$?"
python s46_figures.py; echo "STATUS s46=$?"
python s46_figures_lyc.py; echo "STATUS s46lyc=$?"
echo "ALL_DONE"
