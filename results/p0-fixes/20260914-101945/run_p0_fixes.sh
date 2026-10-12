#!/bin/bash
# P0 data-provenance fixes: re-run affected analyses; all outputs land on NAS.
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
ROOT=/path/to/code
PY=python

{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_p0.txt 2>&1 || true

declare -A ST
run(){ local name="$1" dir="$2" script="$3"; \
  echo "===== START $name $(date) ====="; \
  ( cd "$ROOT/$dir" && $PY "$script" ); local rc=$?; \
  ST[$name]=$rc; echo "===== END $name rc=$rc $(date) ====="; }

run s05  Module1 s05_factor_analysis.py
run s06  Module1 s06_sensitivity.py
run s07  Module1 s07_visualization_new_v3_lyc.py
run s40  Module4 s40_external_data_pipeline.py
run s44f Module4 s44f_final_events.py
run s36  Module3 s36_figures.py

rc=0
{
  for k in s05 s06 s07 s40 s44f s36; do
    echo "$k=${ST[$k]}"
    [ "${ST[$k]}" -ne 0 ] && rc=1
  done
} | tee statuses.txt
echo "ALL_DONE rc=$rc"
exit $rc
