#!/bin/bash
cd "$(dirname "$0")"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; python3 -m pip freeze; } > env.txt 2>&1 || true
python3 r4_traj_temporal_split.py > log 2>&1
echo $? > exit_code
