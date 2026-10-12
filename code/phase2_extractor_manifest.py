import os
import glob
import json
from datetime import datetime
import pyarrow.parquet as pq

# 配置路径（根据你的实际路径修改）
OUTPUT_DIR = "/path/to/project/cohort_v1"
SHARD_DIR = os.path.join(OUTPUT_DIR, "temp_shards")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "processing_manifest.jsonl")

# 1. 扫描现有parquet文件
existing_files = glob.glob(os.path.join(SHARD_DIR, "*.parquet"))
print(f"发现 {len(existing_files)} 个parquet文件")

# 2. 重建manifest条目
entries = []
for pq_path in sorted(existing_files):
    basename = os.path.basename(pq_path)
    zst_name = basename.replace(".parquet", "")  # RC_2015-01.zst.parquet → RC_2015-01.zst
    
    # 快速读取行数
    try:
        pf = pq.ParquetFile(pq_path)
        count = pf.metadata.num_rows
    except:
        count = 0
    
    entry = {
        "file": zst_name,
        "status": "COMPLETED",
        "count": count,
        "timestamp": datetime.now().isoformat(),
        "error": None
    }
    entries.append(entry)
    print(f"✓ {zst_name}: {count:,} 行")

# 3. 写入新manifest
with open(MANIFEST_PATH, 'w') as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"\n✅ Manifest重建完成: {MANIFEST_PATH}")
print(f"   共 {len(entries)} 个文件，下次运行将自动跳过")