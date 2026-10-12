#!/bin/bash
set -euo pipefail
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311

cd /path/to/code/Module3

# Pin environment
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_a2.txt 2>&1 || true

python s32_phenotype_geometry.py
