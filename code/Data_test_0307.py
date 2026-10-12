import os
import pyarrow.parquet as pq

path = "/path/to/project/cohort_v1/final_longitudinal_data_dataset"
file_0 = os.path.join(path,"RC_2009-10.zst.parquet")
file_1 = os.path.join(path,"RS_2009-10.zst.parquet")
pf = pq.ParquetFile(file_0)
pf_1 = pq.ParquetFile(file_1)

# 2. 查看核心元数据
print("=== 基本文件信息 ===")
print(f"总行数: {pf.metadata.num_rows}")  # 总记录数
print(f"分块数: {pf.num_row_groups}")     # 数据分块数（Parquet的行组）

# 3. 查看列信息（重点：列名 + 数据类型）
print("\n=== 列名 & 数据类型 ===")
# 方式1：简洁查看
for idx,name in enumerate(pf.schema_arrow.names):
    dtype = pf.schema_arrow.types[idx]
    print(f"{name}: {dtype}")

# 方式2：详细schema（含nullable等属性）
print("\n=== 完整Schema ===")
print(pf.schema)  # 输出结构化的schema信息

# 4. 查看分区/额外元数据（如果有）
print("\n=== 额外元数据 ===")
print(pf.metadata.metadata)  # 包含pandas索引、压缩等信息