#!/bin/bash
cd "$(dirname "$0")"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311

R4=/path/to/project/module4_results/validation_r4
export R4_OUT=$R4
mkdir -p "$R4"
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; python3 -m pip freeze; } > env_leak.txt 2>&1 || true
python3 r4_leakage_a.py > log_leakage_a 2>&1
echo $? > exit_code_leak
