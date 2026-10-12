#!/bin/bash
cd "$(dirname "$0")"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311

R4=/path/to/project/module4_results/validation_r4
EXT=/path/to/project/module4_results/data/external/processed/epidemiology_annual.csv
SIG=/path/to/project/module4_results/data/digital_signal_annual_diff.csv
SIGADJ=/path/to/project/module4_results/data/digital_signal_annual_band_adjusted.csv
TRAJ=/path/to/project/module2_results/trajectories/user_trajectories.parquet
M1S=/path/to/project/module1_results/statistics
M2S=/path/to/project/module2_results/statistics

export R4_OUT=$R4 R4_EXT=$EXT R4_TRAJ=$TRAJ R4_M1_STAT=$M1S R4_M2_STAT=$M2S
mkdir -p "$R4"
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; python3 -m pip freeze; } > env.txt 2>&1 || true

: > runlog
for step in \
  "citb python3 simulate_citb.py" \
  "ext_naive python3 external_robustness.py $SIG naive" \
  "ext_corrected python3 external_robustness.py $SIGADJ corrected" \
  "trajsplit python3 r4_traj_temporal_split.py" \
  "summary python3 validation_summary.py" ; do
  name=${step%% *}; cmd=${step#* }
  echo "== $name ==" >> runlog
  eval "$cmd" > "log_$name" 2>&1
  echo "$name=$?" >> runlog
done
echo "ALL_DONE" >> runlog
