#!/bin/bash
cd "$(dirname "$0")"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; python3 -m pip freeze; } > env_classify.txt 2>&1 || true
python3 r4_leakage_a_classify.py > log_leakage_classify 2>&1
echo $? > exit_code_classify
