import os
import glob
import gc
import logging
import hashlib
import io
import sys
import shutil
import contextlib
import psutil  # Required for memory safety
from datetime import datetime
from multiprocessing import Pool
import pandas as pd

# HIGH-PERFORMANCE LIBRARIES
import orjson
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from logging import Filter
# ================= USER CONFIGURATION =================
# [CRITICAL] SET THIS TO A FAST LOCAL DISK WITH >200GB FREE SPACE
# Do not use "/" if it is small. Use "/home/tmp" or "/data/tmp"
LOCAL_SCRATCH_DIR = "/path/to/code" #"/data1/qizhixiao/user/temp_data"#"/data1/user/reddit_emotion_disorder/temp_data" #"/var/tmp" 

# IMPORT USER MODULES
# Ensure config.py and utils.py are in the same directory
try:
    from utils import setup_logger, clean_text, init_nlp_tools, clean_text_batch
    from config import PROCESSING_CONFIG, SUICIDE_KEYWORDS, SELF_HARM_KEYWORDS, DATA_LAYOUT
except ImportError:
    print("Error: 'utils.py' or 'config.py' not found. Please ensure they exist.")
    sys.exit(1)

# ================= CONSTANTS & LIMITS =================
NUM_CORES = PROCESSING_CONFIG["cpu_cores"]    # Optimized for your 300GB Server
BATCH_SIZE = 200000                     # Rows per Parquet Chunk
ZST_WINDOW_SIZE = 2**31                # Max window size for ZST
RAM_FILE_LIMIT_BYTES = 55 * 1024**3     # Max file size to put in RAM (8GB)
LOCAL_DISK_MIN_FREE = 200 * 1024**3     # Keep 60GB free on disk at all times

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

presidio_modules = [
    "presidio_analyzer.recognizer_registry",  # 警告的核心来源
    "presidio_analyzer",
    "presidio_anonymizer"
]
for module in presidio_modules:
    logging.getLogger(module).setLevel(logging.ERROR)  # 仅记录ERROR级日志，屏蔽WARNING
    

# ================= GLOBAL SHARED STATE (ZERO-COPY) =================
TARGET_USERS_GLOBAL = None 
NLP_ANALYZER = None
NLP_ANONYMIZER = None
SUICIDE_KEYWORDS_TUPLE = None
SELF_HARM_KEYWORDS_TUPLE = None

# ================= WORKER INITIALIZATION =================
class AddressWarningFilter(Filter):
    """过滤掉Presidio的ADDRESS实体警告消息"""
    def filter(self, record):
        # 精准匹配要过滤的警告内容
        return "Entity ADDRESS doesn't have the corresponding recognizer" not in record.getMessage()
def init_worker():
    """
    Called by each worker process after fork.
    优化点：
    1. 正确屏蔽Presidio日志 + 精准过滤ADDRESS警告
    2. 增强异常处理（日志+堆栈）
    3. 调整GC阈值适配12进程
    4. 提升NLP初始化稳定性
    """
    global NLP_ANALYZER, NLP_ANONYMIZER, SUICIDE_KEYWORDS_TUPLE, SELF_HARM_KEYWORDS_TUPLE
    try:
        # ========== 1. 彻底屏蔽Presidio所有无关日志（核心修复） ==========
        # 1.1 修正logger名称（下划线），屏蔽所有Presidio子模块
        presidio_loggers = [
            "presidio",
            "presidio_analyzer",
            "presidio_analyzer.recognizer_registry",
            "presidio_anonymizer"
        ]
        for logger_name in presidio_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)  # 仅保留CRITICAL级
            logger.propagate = False  # 阻止日志向上传播
            logger.addFilter(AddressWarningFilter())  # 精准过滤ADDRESS警告
        
        # 1.2 兜底：给根日志器也添加过滤器（防止Presidio绕开子日志器）
        root_logger = logging.getLogger()
        root_logger.addFilter(AddressWarningFilter())

        # ========== 2. 加载NLP工具（增强稳定性） ==========
        try:
            _, NLP_ANALYZER, NLP_ANONYMIZER = init_nlp_tools()
        except Exception as nlp_e:
            # 单独捕获NLP初始化错误，更精准
            raise RuntimeError(f"NLP工具初始化失败: {str(nlp_e)}") from nlp_e
        
        # ========== 3. 优化关键词（保留原有逻辑） ==========
        SUICIDE_KEYWORDS_TUPLE = tuple(k.lower() for k in SUICIDE_KEYWORDS)
        SELF_HARM_KEYWORDS_TUPLE = tuple(k.lower() for k in SELF_HARM_KEYWORDS)
        
        # ========== 4. 调整GC阈值（适配12进程，减少内存堆积） ==========
        # 降低阈值，加快临时对象回收，12进程下更稳定
        gc.set_threshold(500, 5, 5) 
        
        # 可选：打印worker初始化成功日志（便于调试）
        logging.info(f"Worker进程初始化成功 (PID:{os.getpid()})")
        
    except Exception as e:
        # ========== 5. 增强异常处理（日志+堆栈+stderr） ==========
        error_msg = f"Worker Init Error (PID:{os.getpid()}): {str(e)}"
        # 1. 写入stderr（保留原有逻辑）
        sys.stderr.write(f"{error_msg}\n")
        # 2. 记录到日志（带完整堆栈，便于调试）
        logging.error(error_msg, exc_info=True)
        # 3. 抛出异常，终止该worker进程（避免无效进程运行）
        raise
    
# ================= SMART FILE MANAGER (INPUT) =================
@contextlib.contextmanager
def smart_open_zst(zst_path_nas):
    """
    Determines the best way to read the input file based on size and resources.
    - Tier 1: RAM Copy (/dev/shm) - Fastest (for files < 8GB)
    - Tier 2: Local Disk Copy - Fast (for files < 32GB)
    - Tier 3: NAS Stream - Slow Fallback (if no space)
    """
    file_name = os.path.basename(zst_path_nas)
    try:
        file_size = os.path.getsize(zst_path_nas)
    except OSError:
        yield zst_path_nas, "NAS_STREAM_ERROR"
        return

    pid = os.getpid()
    temp_path = None
    strategy = "NAS_STREAM" # Default fallback
    
    try:
        # CHECK 1: RAM Copy (/dev/shm)
        dev_shm_stats = psutil.disk_usage("/dev/shm")
        dev_shm_free = dev_shm_stats.free
        
        dev_shm_sys_reserve = 7 * 1024**3 
        dev_shm_per_core = (dev_shm_free - dev_shm_sys_reserve) / NUM_CORES
        
        if file_size <= RAM_FILE_LIMIT_BYTES and dev_shm_free > (file_size + 10 * 1024**3):
            temp_path = os.path.join("/dev/shm", f"proc_{pid}_{file_name}")
            strategy = "RAM_COPY"
        # CHECK 2: Local Disk Copy (NVMe/SSD)
        if strategy == "NAS_STREAM":
            try:
                os.makedirs(LOCAL_SCRATCH_DIR, exist_ok=True)
                disk = shutil.disk_usage(LOCAL_SCRATCH_DIR)
                # Safety: Ensure we don't fill the disk completely
                if disk.free > (file_size + LOCAL_DISK_MIN_FREE):
                    temp_path = os.path.join(LOCAL_SCRATCH_DIR, f"proc_{pid}_{file_name}")
                    strategy = "DISK_COPY"
            except Exception:
                pass 

        # EXECUTE COPY STRATEGY
        if strategy != "NAS_STREAM" and temp_path:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                # Copy from NAS to Local
                shutil.copy2(zst_path_nas, temp_path)
                yield temp_path, strategy
            except IOError:
                # Copy failed (e.g. disk full mid-copy), fallback to NAS
                yield zst_path_nas, "NAS_STREAM_FALLBACK"
        else:
            # Fallback: Read directly from NAS
            yield zst_path_nas, "NAS_STREAM"

    finally:
        # CLEANUP: Always delete the input temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass 

# ================= 【修改点6：新增批量清理函数，解决12进程残留】 =================
def batch_cleanup_temp_files():
    """主进程结束后，批量清理12进程残留的临时文件"""
    marker_file = "/dev/shm/zst_cleanup_marker.txt"
    if os.path.exists(marker_file):
        with open(marker_file, "r") as f:
            temp_paths = [p.strip() for p in f.readlines()]
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"批量清理残留文件：{path}")
                except OSError as e:
                    logger.error(f"批量清理{path}失败：{e}")
        os.remove(marker_file)
    
    # 清理LOCAL_SCRATCH_DIR中残留的out_*文件
    local_out_pattern = os.path.join(LOCAL_SCRATCH_DIR, "out_*.parquet")
    for out_file in glob.glob(local_out_pattern):
        try:
            os.remove(out_file)
            logger.debug(f"清理残留输出文件：{out_file}")
        except OSError as e:
            logger.warning(f"清理{out_file}失败：{e}")
            
# ================= CORE EXTRACTOR =================
def extract_file_optimized(args):
    zst_path, output_shard_dir_nas = args
    file_name = os.path.basename(zst_path)
    
    # OUTPUT STRATEGY: Write Local -> Move to NAS
    # This prevents network latency ("chattiness") during Parquet creation
    pid = os.getpid()
    local_output_filename = f"{file_name}.parquet"
    local_output_path = os.path.join(LOCAL_SCRATCH_DIR, f"out_{pid}_{local_output_filename}")
    final_nas_path = os.path.join(output_shard_dir_nas, local_output_filename)

    # Global State Access
    target_users = TARGET_USERS_GLOBAL
    suicide_keywords = SUICIDE_KEYWORDS_TUPLE
    self_harm_keywords = SELF_HARM_KEYWORDS_TUPLE
    
    records_buffer = []
    
    raw_text_buffer = []
    total_count = 0
    writer = None
    
    try:
        is_submission = file_name.startswith("RS_")
        
        # 1. Open Input (Smart Staging)
        with smart_open_zst(zst_path) as (process_path, strategy):
            
            # If streaming from NAS, use massive buffer (128MB) to reduce IOPS
            io_buffer = 128 * 1024 * 1024 if "NAS" in strategy else -1
            
            with open(process_path, 'rb', buffering=io_buffer) as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=ZST_WINDOW_SIZE)
                
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                    
                    for line in text_stream:
                        if not line: 
                            continue
                        # ========== 新增：提前过滤无效用户（核心优化） ==========
                        try:
                            # 快速提取author字段（无需完整解析JSON，快10倍）
                            if '"author":"' not in line:
                                continue
                            author_start = line.index('"author":"') + len('"author":"')
                            author_end = line.index('"', author_start)
                            author = line[author_start:author_end]
                            # 提前过滤非目标用户/删除的用户
                            if author not in target_users or author == "[deleted]":
                                continue
                        except:
                            continue

                        try:
                            # Fast JSON Parse
                            obj = orjson.loads(line)
                        except:
                            continue

                        # Fast Membership Check
                        author = obj.get("author")
                        if not author or author == "[deleted]": 
                            continue                            
                        # Dict lookup O(1)
                        if author not in target_users:
                            continue
                        
                        # --- DATA NORMALIZATION ---
                        if is_submission:
                            title = obj.get("title", "")
                            selftext = obj.get("selftext", "")
                            if selftext in ["[deleted]", "[removed]", None]: selftext = ""
                            raw_text = f"{title}\n{selftext}".strip()
                            data_type = "submission"
                            
                            link_id = obj.get("id")
                            if link_id and not link_id.startswith("t3_"): link_id = f"t3_{link_id}"
                            parent_id = None
                            is_top_level = True
                            interaction_type = "root_story"
                            domain = obj.get("domain", "")
                            url = obj.get("url", "")
                            num_comments = obj.get("num_comments", 0)
                            controversiality = 0
                        else:
                            raw_text = obj.get("body", "").strip()
                            data_type = "comment"
                            
                            parent_id = obj.get("parent_id", "")
                            link_id = obj.get("link_id", "")
                            
                            if parent_id and (parent_id.startswith("t3_") or parent_id == link_id):
                                is_top_level = True
                                interaction_type = "direct_reply_to_op"
                            else:
                                is_top_level = False
                                interaction_type = "nested_discussion"
                            domain = ""
                            url = ""
                            num_comments = 0
                            controversiality = obj.get("controversiality", 0)

                        if not raw_text or raw_text in ["[deleted]", "[removed]"]:
                            continue

                        # --- METADATA & NLP ---
                        user_info = target_users[author]
                        post_ts = int(obj.get("created_utc", 0))
                        
                        if user_info["group"] == "patient":
                            t0_date = user_info["t0_date"]
                            is_diag = (obj.get("id") == user_info.get("diagnosis_post_id"))
                            rel_days = round((post_ts - user_info["t0_ts"]) / 86400, 2)
                        else:
                            t0_date = "PENDING"
                            is_diag = False
                            rel_days = 0.0

                        # Optimized Keyword Search (Short-circuit loop)
                        txt_low = raw_text.lower()
                        flag_suicide = any(k in txt_low for k in suicide_keywords) 
                        flag_self_harm = any(k in txt_low for k in self_harm_keywords)
                        
                        # 原逻辑：clean_text_content = clean_text(raw_text, NLP_ANALYZER, NLP_ANONYMIZER)
                        # 新逻辑：先缓存raw_text，批量清理
                        raw_text_buffer.append(raw_text)

                        # --- APPEND RECORD ---
                        records_buffer.append({
                            "user_id_hash": hashlib.sha256((str(author) + PROCESSING_CONFIG["salt"]).encode()).hexdigest(),
                            "post_timestamp": post_ts,
                            "post_datetime": datetime.utcfromtimestamp(post_ts).strftime("%Y-%m-%d %H:%M:%S"),
                            "relative_days": rel_days,
                            "data_type": data_type,
                            "interaction_type": interaction_type,
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
                            "has_media": bool(obj.get("media") or (url and url.endswith(('.jpg', '.png', '.gif')))),
                            "domain": domain,
                            "author_flair": obj.get("author_flair_text") or "",
                            "clean_text": "",
                            "flag_suicide": flag_suicide,
                            "flag_self_harm": flag_self_harm,
                            "group": user_info["group"],
                            "disorder_group": user_info["disorder"],
                            "t0_date": t0_date,
                            "is_diagnosis_post": is_diag,
                            "is_retrospective": user_info.get("is_retrospective", False),
                            "post_id": obj.get("id", "")
                        })

                        # --- FLUSH BATCH (TO LOCAL DISK) ---
                        if len(records_buffer) >= BATCH_SIZE:
                            # ========== 新增：批量清理文本 ==========
                            clean_texts = clean_text_batch(
                                raw_text_buffer, 
                                NLP_ANALYZER, 
                                NLP_ANONYMIZER,
                                batch_size=1000  # 每1000条一批
                            )
                            for idx, record in enumerate(records_buffer):
                                record["clean_text"] = clean_texts[idx]
                                
                            table = pa.Table.from_pylist(records_buffer)
                            if writer is None:
                                # Write to LOCAL scratch path first
                                writer = pq.ParquetWriter(local_output_path, table.schema, compression='snappy')
                            writer.write_table(table)
                            total_count += len(records_buffer)
                            
                            del clean_texts, table
                            records_buffer.clear()  # 用clear()而非重新赋值，减少GC压力
                            raw_text_buffer.clear()
                            gc.collect()  # 批量清理后手动触发GC
                            
    except Exception as e:
        if writer: 
            writer.close()
            del writer
        if os.path.exists(local_output_path): 
            os.remove(local_output_path)
        return {'file': file_name, 'status': 'FAILED', 'error': str(e)}

    # Final Buffer Flush
    if records_buffer:
        clean_texts = clean_text_batch(raw_text_buffer, NLP_ANALYZER, NLP_ANONYMIZER)
        for idx, record in enumerate(records_buffer):
            record["clean_text"] = clean_texts[idx]
            
        table = pa.Table.from_pylist(records_buffer)
        if writer is None:
            writer = pq.ParquetWriter(local_output_path, table.schema, compression='snappy')
        writer.write_table(table)
        total_count += len(records_buffer)
        
        del clean_texts, table
        records_buffer.clear()
        raw_text_buffer.clear()

    if writer: 
        writer.close()
        del writer
    # --- CRITICAL: MOVE TO NAS ---
    # We move the completed file in one sequential operation
    if total_count > 0 and os.path.exists(local_output_path):
        try:
            shutil.move(local_output_path, final_nas_path)
            return {'file': file_name, 'status': 'COMPLETED', 'count': total_count}
        except Exception as move_error:
            # Retry with copy+delete if move fails (e.g. cross-filesystem)
            try:
                shutil.copy2(local_output_path, final_nas_path)
                os.remove(local_output_path)
                return {'file': file_name, 'status': 'COMPLETED', 'count': total_count}
            except Exception as copy_error:
                return {'file': file_name, 'status': 'FAILED_MOVE', 'error': str(copy_error)}
    else:
        # Empty file
        if os.path.exists(local_output_path): 
            os.remove(local_output_path)
        return {'file': file_name, 'status': 'EMPTY', 'count': 0}

# ================= MASTER CONTROLLER =================
def extract_user_history(zst_root_dir, master_index_path):
    global TARGET_USERS_GLOBAL

    # 1. LOAD INDEX (Parent Process)
    logger.info(f"Loading Master Index from {master_index_path}...")
    master_df = pd.read_parquet(master_index_path)
    TARGET_USERS_GLOBAL = master_df.set_index("user_id").to_dict("index")
    logger.info(f"Target Users Loaded: {len(TARGET_USERS_GLOBAL):,} in memory.")

    # 2. SETUP PATHS
    output_dir = PROCESSING_CONFIG["output_dir"]
    shard_dir_nas = os.path.join(output_dir, "temp_shards") # NAS Destination
    os.makedirs(shard_dir_nas, exist_ok=True)
    os.makedirs(LOCAL_SCRATCH_DIR, exist_ok=True)
    
    # 3. MANIFEST (Resume Support)
    manifest_path = os.path.join(output_dir, "processing_manifest.jsonl")
    processed_files = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            for line in f:
                try:
                    entry = orjson.loads(line)
                    if entry['status'] in ['COMPLETED', 'EMPTY']:
                        processed_files.add(entry['file'])
                except: pass

    # 4. GATHER FILES
    comments_path = os.path.join(zst_root_dir, DATA_LAYOUT["comments_dir"], "RC_*.zst")
    submissions_path = os.path.join(zst_root_dir, DATA_LAYOUT["submissions_dir"], "RS_*.zst")
    all_files = sorted(glob.glob(comments_path) + glob.glob(submissions_path))
    
    worklist = []
    for f in all_files:
        if os.path.basename(f) not in processed_files:
            worklist.append((f, shard_dir_nas))
            
    logger.info(f"Total Files: {len(all_files)}. Left to Process: {len(worklist)}")

    # 5. START POOL
    # Open Manifest in append mode to log real-time
    with open(manifest_path, 'ab') as manifest_file:
        with Pool(NUM_CORES, initializer=init_worker) as pool:
            pbar = tqdm(total=len(worklist), desc="Extracting", unit="file")
            
            for result in pool.imap_unordered(extract_file_optimized, worklist):
                # Log to Manifest
                record = {
                    "file": result['file'],
                    "status": result['status'],
                    "count": result.get('count', 0),
                    "timestamp": datetime.now().isoformat(),
                    "error": result.get('error')
                }
                manifest_file.write(orjson.dumps(record) + b"\n")
                manifest_file.flush()
                
                # Log Errors
                if result['status'] in ['FAILED', 'FAILED_MOVE']:
                    logger.error(f"FAIL: {result['file']} - {result.get('error')}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "Last": result['file'], 
                    "Rows": f"{result.get('count', 0):,}"
                })
            
            pbar.close()
    batch_cleanup_temp_files()
    logger.info("All files processed, batch cleanup completed.")
    
if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst_root_dir", required=True, help="Root folder containing RC_ and RS_ folders")
    parser.add_argument("--master_index", required=True, help="Path to patient/control parquet index")
    args = parser.parse_args()
    
    # Enforce Fork for Zero-Copy memory efficiency
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass 

    extract_user_history(args.zst_root_dir, args.master_index)