# retry_failed_phase35.py
import os
import sys
import pandas as pd
from post_processing import build_final_dataset, init_final_build_worker, process_final_build_chunk
from multiprocessing import Pool

# 配置
FAILED_FILES = [
    "RC_2021-05.zst.parquet"
    # "RC_2020-05.zst.parquet",
    # "RC_2020-07.zst.parquet", 
    # "RC_2021-01.zst.parquet",
    # "RC_2021-03.zst.parquet"
]

BASE_DIR = "/path/to/project/cohort_v1"
SHARD_DIR = os.path.join(BASE_DIR, "temp_shards")
FINAL_DIR = os.path.join(BASE_DIR, "final_longitudinal_data")
MATCHED_META_PATH = os.path.join(BASE_DIR, "intermediate_results/step_3.4_psm_matched.parquet")

# 加载匹配元数据
print("加载PSM匹配结果...")
matched_meta = pd.read_parquet(MATCHED_META_PATH)
user_map = matched_meta.set_index("user_id_hash")["shadow_t0_ts"].to_dict()
print(f"目标用户: {len(user_map):,}人")

# 构建工作列表
work_args = []
for fname in FAILED_FILES:
    shard_path = os.path.join(SHARD_DIR, fname)
    output_path = os.path.join(FINAL_DIR, fname)
    
    if not os.path.exists(shard_path):
        print(f"❌ 源文件不存在: {fname}")
        continue
    
    # 验证源文件
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(shard_path)
        print(f"✓ {fname}: 源文件有效 ({pf.metadata.num_rows:,}行)")
        work_args.append((shard_path, user_map, output_path))
    except Exception as e:
        print(f"❌ {fname}: 源文件损坏 - {e}")

print(f"\n准备重跑: {len(work_args)}个文件")

# 串行处理（避免NAS压力）
from tqdm import tqdm
results = []
for args in tqdm(work_args, desc="Processing failed files"):
    result = process_final_build_chunk(args)
    results.append(result)

print(f"\n完成: {sum(results)}/{len(results)} 成功")