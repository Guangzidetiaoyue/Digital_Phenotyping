#!/bin/bash
cd "$(dirname "$0")"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
{ python3 -V; python3 -m pip freeze; } > env_gmm.txt 2>&1 || true
python3 r4_gmm_robustness.py > log_gmm 2>&1
echo $? > exit_code_gmm
