#!/bin/bash
set -euo pipefail
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
cd /path/to/code/Module3
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_s31c.txt 2>&1 || true
python s31c_external_or_sensitivity.py
echo "EXIT_OK"
