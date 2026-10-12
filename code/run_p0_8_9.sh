#!/bin/bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
ROOT=/path/to/code
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_p089.txt 2>&1 || true

cd "$ROOT/Module3" && python s33_disorder_networks.py; echo "STATUS s33=$?"
cd "$ROOT/Module4" && python s46_figures.py; echo "STATUS s46=$?"
echo "ALL_DONE"
