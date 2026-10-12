# fix_rc_2021_05_v2.py
import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import zstandard as zstd
import io
import json
import hashlib
from datetime import datetime
import shutil
import pyarrow as pa
# 配置路径
ZST_ROOT = "/mnt/path/to/reddit"
SHARD_DIR = "/path/to/project/cohort_v1/temp_shards"
MASTER_INDEX = "/path/to/project/cohort_v1/master_patient_index.parquet"
OUTPUT_DIR = "/path/to/project/cohort_v1"

# 加载目标用户
print("加载目标用户...")
master_df = pd.read_parquet(MASTER_INDEX)
target_users = master_df.set_index("user_id").to_dict("index")
print(f"加载了 {len(target_users):,} 个目标用户")

# 配置
BATCH_SIZE = 200000
SALT = "NATURE_PAPER_2025_SECURE_SALT_v9X"  # 与config.py一致

# 关键词（简化版，仅检查）
SUICIDE_KEYWORDS = ["suicide", "suicidal", "kill myself", "end my life"]
SELF_HARM_KEYWORDS = ["self harm", "cutting", "cut myself"]

def clean_text_fast(text):
    """简化的文本清洗"""
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_single_file(zst_path, output_dir, target_users_dict):
    """独立提取函数，不依赖全局变量"""
    file_name = os.path.basename(zst_path)
    output_path = os.path.join(output_dir, f"{file_name}.parquet")
    
    print(f"处理: {file_name}")
    
    records_buffer = []
    total_count = 0
    writer = None
    
    try:
        is_submission = file_name.startswith("RS_")
        
        with open(zst_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                
                for line in text_stream:
                    if not line:
                        continue
                    
                    # 快速过滤：检查author字段
                    if '"author":"' not in line:
                        continue
                    
                    # 提取author
                    try:
                        author_start = line.index('"author":"') + 10
                        author_end = line.index('"', author_start)
                        author = line[author_start:author_end]
                        if author not in target_users_dict or author == "[deleted]":
                            continue
                    except:
                        continue
                    
                    # 解析JSON
                    try:
                        obj = json.loads(line)
                    except:
                        continue
                    
                    author = obj.get("author")
                    if not author or author not in target_users_dict:
                        continue
                    
                    # 提取数据（与原版一致）
                    if is_submission:
                        title = obj.get("title", "")
                        selftext = obj.get("selftext", "")
                        if selftext in ["[deleted]", "[removed]", None]: 
                            selftext = ""
                        raw_text = f"{title}\n{selftext}".strip()
                        data_type = "submission"
                        parent_id = None
                        link_id = obj.get("id")
                        is_top_level = True
                        domain = obj.get("domain", "")
                        num_comments = obj.get("num_comments", 0)
                        controversiality = 0
                    else:
                        raw_text = obj.get("body", "").strip()
                        data_type = "comment"
                        parent_id = obj.get("parent_id", "")
                        link_id = obj.get("link_id", "")
                        is_top_level = parent_id and (parent_id.startswith("t3_") or parent_id == link_id)
                        domain = ""
                        num_comments = 0
                        controversiality = obj.get("controversiality", 0)
                    
                    if not raw_text or raw_text in ["[deleted]", "[removed]"]:
                        continue
                    
                    # 元数据
                    user_info = target_users_dict[author]
                    post_ts = int(obj.get("created_utc", 0))
                    
                    if user_info["group"] == "patient":
                        t0_date = user_info["t0_date"]
                        is_diag = (obj.get("id") == user_info.get("diagnosis_post_id"))
                        rel_days = round((post_ts - user_info["t0_ts"]) / 86400, 2)
                    else:
                        t0_date = "PENDING"
                        is_diag = False
                        rel_days = 0.0
                    
                    # 关键词标记
                    txt_low = raw_text.lower()
                    flag_suicide = any(k in txt_low for k in SUICIDE_KEYWORDS)
                    flag_self_harm = any(k in txt_low for k in SELF_HARM_KEYWORDS)
                    
                    # 清洗文本
                    clean_txt = clean_text_fast(raw_text)
                    
                    records_buffer.append({
                        "user_id_hash": hashlib.sha256((str(author) + SALT).encode()).hexdigest(),
                        "post_timestamp": post_ts,
                        "post_datetime": datetime.utcfromtimestamp(post_ts).strftime("%Y-%m-%d %H:%M:%S"),
                        "relative_days": rel_days,
                        "data_type": data_type,
                        "interaction_type": "root_story" if is_submission else ("direct_reply_to_op" if is_top_level else "nested_discussion"),
                        "parent_id": parent_id,
                        "link_id": link_id,
                        "is_top_level": is_top_level,
                        "num_comments": num_comments,
                        "subreddit": obj.get("subreddit", ""),
                        "score": obj.get("score", 0),
                        "controversiality": controversiality,
                        "gilded": obj.get("gilded", 0),
                        "is_edited": obj.get("edited") is not False,
                        "is_nsfw": obj.get("over_18", False),
                        "has_media": bool(obj.get("media")),
                        "domain": domain,
                        "author_flair": obj.get("author_flair_text") or "",
                        "clean_text": clean_txt,
                        "flag_suicide": flag_suicide,
                        "flag_self_harm": flag_self_harm,
                        "group": user_info["group"],
                        "disorder_group": user_info["disorder"],
                        "t0_date": t0_date,
                        "is_diagnosis_post": is_diag,
                        "is_retrospective": user_info.get("is_retrospective", False),
                        "post_id": obj.get("id", "")
                    })
                    
                    if len(records_buffer) >= BATCH_SIZE:
                        table = pa.Table.from_pylist(records_buffer)
                        if writer is None:
                            import pyarrow.parquet as pq
                            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                        writer.write_table(table)
                        total_count += len(records_buffer)
                        records_buffer.clear()
                        print(f"  已处理 {total_count:,} 行")
        
        # 最后刷新
        if records_buffer:
            table = pa.Table.from_pylist(records_buffer)
            if writer is None:
                import pyarrow.parquet as pq
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
            total_count += len(records_buffer)
        
        if writer:
            writer.close()
        
        print(f"✅ 完成: {file_name} -> {total_count:,} 行")
        return {'file': file_name, 'status': 'COMPLETED', 'count': total_count}
        
    except Exception as e:
        print(f"❌ 失败: {file_name} - {str(e)}")
        import traceback
        traceback.print_exc()
        return {'file': file_name, 'status': 'FAILED', 'error': str(e)}

# 执行
if __name__ == "__main__":
    zst_file = os.path.join(ZST_ROOT, "comments", "RC_2021-05.zst")
    
    if not os.path.exists(zst_file):
        print(f"ZST文件不存在: {zst_file}")
        sys.exit(1)
    
    result = extract_single_file(zst_file, SHARD_DIR, target_users)
    
    print(f"\n结果: {result}")
    
    if result['status'] == 'COMPLETED':
        print(f"\n现在可以运行Phase 3.5处理该文件...")
        print(f"python -c \"from post_processing import process_final_build_chunk; ...\"")