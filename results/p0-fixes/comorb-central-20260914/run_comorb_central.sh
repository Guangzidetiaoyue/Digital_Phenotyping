#!/bin/bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate my_py311
ROOT=/path/to/code
{ python3 -V; echo "PLATFORM=$(uname -s)-$(uname -m)"; \
  echo '--- pip freeze ---'; python3 -m pip freeze; } > env_comorb.txt 2>&1 || true

cd "$ROOT/Module3"
python s31_pairwise_distinguishability.py; echo "STATUS s31=$?"
python s31c_external_or_sensitivity.py; echo "STATUS s31c=$?"
python s31b_empirical_comorbidity.py; echo "STATUS s31b=$?"
python s32_phenotype_geometry.py; echo "STATUS s32=$?"
python s36_figures.py; echo "STATUS s36=$?"
python s36_figures_lyc.py; echo "STATUS s36lyc=$?"
echo "ALL_DONE"
