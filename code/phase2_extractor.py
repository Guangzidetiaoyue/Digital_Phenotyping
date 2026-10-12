import os
import glob
import gc
import logging
import hashlib
import io
import sys
import shutil
import contextlib
import psutil
from datetime import datetime
from multiprocessing import Pool
import pandas as pd

# HIGH-PERFORMANCE LIBRARIES
import orjson,json
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from logging import Filter

# ================= USER CONFIGURATION =================
# [CRITICAL] LOCAL SCRATCH PATH
# Fallback disk path if RAM (/dev/shm) is full.
LOCAL_SCRATCH_DIR = "/path/to/code" #"/data1/qizhixiao/user/temp_data"

# IMPORT USER MODULES
try:
    # Added clean_text_fast here
    from utils import setup_logger, clean_text_fast, init_nlp_tools
    from config import PROCESSING_CONFIG, SUICIDE_KEYWORDS, SELF_HARM_KEYWORDS, DATA_LAYOUT
except ImportError:
    print("Error: 'utils.py' or 'config.py' not found. Please ensure they exist.")
    sys.exit(1)

# ================= CONSTANTS & LIMITS =================
# Set to 80 to utilize your 96 cores effectively while leaving room for OS
NUM_CORES = PROCESSING_CONFIG["cpu_cores"] 
BATCH_SIZE = 200000                     
ZST_WINDOW_SIZE = 2147483648            # 2GB Window
RAM_FILE_LIMIT_BYTES = 55 * 1024**3     # Max file size to put in RAM (55GB)
LOCAL_DISK_MIN_FREE = 100 * 1024**3     # Keep 100GB free on disk

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================= GLOBAL SHARED STATE (ZERO-COPY) =================
TARGET_USERS_GLOBAL = None 
SUICIDE_KEYWORDS_TUPLE = None
SELF_HARM_KEYWORDS_TUPLE = None

# ================= WORKER INITIALIZATION =================
class AddressWarningFilter(Filter):
    """Filter Presidio ADDRESS warnings"""
    def filter(self, record):
        return "Entity ADDRESS doesn't have the corresponding recognizer" not in record.getMessage()

def init_worker():
    """
    Called by each worker process after fork.
    Optimized: Removed heavy NLP loading. Only Regex is used in Phase 2.
    """
    global SUICIDE_KEYWORDS_TUPLE, SELF_HARM_KEYWORDS_TUPLE
    try:
        # 1. Suppress Presidio Logs (Just in case)
        presidio_loggers = ["presidio", "presidio_analyzer", "presidio_anonymizer"]
        for logger_name in presidio_loggers:
            l = logging.getLogger(logger_name)
            l.setLevel(logging.CRITICAL)
            l.propagate = False

        # 2. Optimize Keywords
        SUICIDE_KEYWORDS_TUPLE = tuple(k.lower() for k in SUICIDE_KEYWORDS)
        SELF_HARM_KEYWORDS_TUPLE = tuple(k.lower() for k in SELF_HARM_KEYWORDS)
        
        # 3. GC Tuning (Aggressive for speed)
        gc.set_threshold(700, 10, 10) 
        
    except Exception as e:
        error_msg = f"Worker Init Error (PID:{os.getpid()}): {str(e)}"
        sys.stderr.write(f"{error_msg}\n")
        logging.error(error_msg, exc_info=True)
        raise
    
# ================= SMART FILE MANAGER (YOUR LOGIC KEPT) =================
@contextlib.contextmanager
def smart_open_zst(zst_path_nas):
    """
    Determines the best way to read the input file based on size and resources.
    Uses your logic to check /dev/shm specifically.
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
        # Calculates Per-Core availability to be safe on shared server
        dev_shm_stats = psutil.disk_usage("/dev/shm")
        dev_shm_free = dev_shm_stats.free
        
        dev_shm_sys_reserve = 10 * 1024**3 # Reserve 10GB for system
        dev_shm_per_core = (dev_shm_free - dev_shm_sys_reserve) / NUM_CORES
        
        # Logic: If total free RAM is huge, allows big files. 
        # If running many cores, ensures we don't OOM.
        if file_size <= RAM_FILE_LIMIT_BYTES and dev_shm_free > (file_size + 20 * 1024**3):
            temp_path = os.path.join("/dev/shm", f"proc_{pid}_{file_name}")
            strategy = "RAM_COPY"
            
        # CHECK 2: Local Disk Copy (NVMe/SSD)
        if strategy == "NAS_STREAM":
            try:
                os.makedirs(LOCAL_SCRATCH_DIR, exist_ok=True)
                disk = shutil.disk_usage(LOCAL_SCRATCH_DIR)
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
                shutil.copy2(zst_path_nas, temp_path)
                yield temp_path, strategy
            except IOError:
                yield zst_path_nas, "NAS_STREAM_FALLBACK"
        else:
            yield zst_path_nas, "NAS_STREAM"

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass 

def batch_cleanup_temp_files():
    """Cleanup logic"""
    # Clean /dev/shm
    for f in glob.glob("/dev/shm/proc_*"):
        try: os.remove(f)
        except: pass
    
    # Clean Local Scratch
    local_out_pattern = os.path.join(LOCAL_SCRATCH_DIR, "out_*.parquet")
    for out_file in glob.glob(local_out_pattern):
        try:
            os.remove(out_file)
        except OSError as e:
            logger.warning(f"Cleanup failed: {e}")
            
# ================= CORE EXTRACTOR (OPTIMIZED) =================
def extract_file_optimized(args):
    zst_path, output_shard_dir_nas = args
    file_name = os.path.basename(zst_path)
    
    pid = os.getpid()
    local_output_filename = f"{file_name}.parquet"
    local_output_path = os.path.join(LOCAL_SCRATCH_DIR, f"out_{pid}_{local_output_filename}")
    final_nas_path = os.path.join(output_shard_dir_nas, local_output_filename)
    
    # Skip if done
    if os.path.exists(final_nas_path):
        return {'file': file_name, 'status': 'SKIPPED', 'count': 0}

    # Global State Access
    target_users = TARGET_USERS_GLOBAL
    suicide_keywords = SUICIDE_KEYWORDS_TUPLE
    self_harm_keywords = SELF_HARM_KEYWORDS_TUPLE
    
    records_buffer = []
    total_count = 0
    writer = None
    
    try:
        is_submission = file_name.startswith("RS_")
        
        with smart_open_zst(zst_path) as (process_path, strategy):
            
            # Massive buffer for NAS reading
            io_buffer = 128 * 1024 * 1024 if "NAS" in strategy else -1
            
            with open(process_path, 'rb', buffering=io_buffer) as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=ZST_WINDOW_SIZE)
                
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                    
                    for line in text_stream:
                        if not line: continue
                        
                        # --- OPTIMIZATION: Fast Filter ---
                        # Skip full JSON parse if author isn't even in the line
                        if '"author":"' not in line: continue
                        
                        try:
                            author_start = line.index('"author":"') + 10
                            author_end = line.index('"', author_start)
                            if author_end <= author_start:  # 空或异常
                                continue
                            author = line[author_start:author_end]
                            
                            if author not in target_users or author == "[deleted]":
                                continue
                            if len(author) > 30:  # Reddit用户名通常<20字符
                                continue
                            
                        except ValueError:  # 子串未找到
                            continue
                        # Full Parse
                        try:
                            obj = orjson.loads(line)
                        except:
                            continue

                        # Double check (strictly required)
                        author = obj.get("author")
                        if not author or author not in target_users:
                            continue
                        
                        # --- DATA EXTRACTION ---
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

                        # --- METADATA ---
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

                        # Keyword Search
                        txt_low = raw_text.lower()
                        flag_suicide = any(k in txt_low for k in suicide_keywords) 
                        flag_self_harm = any(k in txt_low for k in self_harm_keywords)
                        
                        # --- CRITICAL OPTIMIZATION ---
                        # REPLACED: clean_text_batch(raw_text_buffer...)
                        # WITH: clean_text_fast (Regex Only)
                        # This removes the CPU bottleneck.
                        clean_txt = clean_text_fast(raw_text)

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
                            "clean_text": clean_txt, # Fast Cleaned
                            "flag_suicide": flag_suicide,
                            "flag_self_harm": flag_self_harm,
                            "group": user_info["group"],
                            "disorder_group": user_info["disorder"],
                            "t0_date": t0_date,
                            "is_diagnosis_post": is_diag,
                            "is_retrospective": user_info.get("is_retrospective", False),
                            "post_id": obj.get("id", "")
                        })

                        # --- FLUSH BATCH ---
                        if len(records_buffer) >= BATCH_SIZE:
                            table = pa.Table.from_pylist(records_buffer)
                            if writer is None:
                                writer = pq.ParquetWriter(local_output_path, table.schema, compression='snappy')
                            writer.write_table(table)
                            total_count += len(records_buffer)
                            records_buffer.clear()
                            
    except Exception as e:
        if writer: writer.close()
        if os.path.exists(local_output_path): os.remove(local_output_path)
        return {'file': file_name, 'status': 'FAILED', 'error': str(e)}

    # Final Buffer Flush
    if records_buffer:
        table = pa.Table.from_pylist(records_buffer)
        if writer is None:
            writer = pq.ParquetWriter(local_output_path, table.schema, compression='snappy')
        writer.write_table(table)
        total_count += len(records_buffer)
        records_buffer.clear()

    if writer: writer.close()

    # Move to NAS
    if total_count > 0 and os.path.exists(local_output_path):
        try:
            # 1. 使用 copyfile 而不是 move/copy2
            # copyfile 只复制文件内容，不复制权限/时间戳等元数据，避免 NAS 权限报错
            shutil.copyfile(local_output_path, final_nas_path)
            
            # 2. 确保写入到 NAS 成功后，再删除本地临时文件
            os.remove(local_output_path)
            
            return {'file': file_name, 'status': 'COMPLETED', 'count': total_count}
        except Exception as e:
            # 如果依然失败，记录详细错误
            return {'file': file_name, 'status': 'FAILED_MOVE', 'error': str(e)}
    else:
        if os.path.exists(local_output_path): os.remove(local_output_path)
        return {'file': file_name, 'status': 'EMPTY', 'count': 0}

# ================= MASTER CONTROLLER =================
def extract_user_history(zst_root_dir, master_index_path):
    global TARGET_USERS_GLOBAL

    logger.info(f"Loading Master Index from {master_index_path}...")
    master_df = pd.read_parquet(master_index_path)
    TARGET_USERS_GLOBAL = master_df.set_index("user_id").to_dict("index")
    logger.info(f"Target Users Loaded: {len(TARGET_USERS_GLOBAL):,} in memory.")

    output_dir = PROCESSING_CONFIG["output_dir"]
    shard_dir_nas = os.path.join(output_dir, "temp_shards") 
    os.makedirs(shard_dir_nas, exist_ok=True)
    os.makedirs(LOCAL_SCRATCH_DIR, exist_ok=True)
    
    # Manifest Loading
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


    # Gather Files
    comments_path = os.path.join(zst_root_dir, DATA_LAYOUT["comments_dir"], "RC_*.zst")
    submissions_path = os.path.join(zst_root_dir, DATA_LAYOUT["submissions_dir"], "RS_*.zst")
    all_files = sorted(glob.glob(comments_path) + glob.glob(submissions_path))
    
    worklist = []
    for f in all_files:
        if os.path.basename(f) not in processed_files:
            worklist.append((f, shard_dir_nas))
            
    logger.info(f"Total Files: {len(all_files)}. Left to Process: {len(worklist)}")

    # Start Pool
    with open(manifest_path, 'ab') as manifest_file:
        with Pool(NUM_CORES, initializer=init_worker) as pool:
            pbar = tqdm(total=len(worklist), desc="Extracting", unit="file")
            
            for result in pool.imap_unordered(extract_file_optimized, worklist):
                # Log
                record = {
                    "file": result['file'],
                    "status": result['status'],
                    "count": result.get('count', 0),
                    "timestamp": datetime.now().isoformat(),
                    "error": result.get('error')
                }
                manifest_file.write(orjson.dumps(record) + b"\n")
                manifest_file.flush()
                
                if result['status'] in ['FAILED', 'FAILED_MOVE']:
                    logger.error(f"FAIL: {result['file']} - {result.get('error')}")
                
                pbar.update(1)
                pbar.set_postfix({"Last": result['file'], "Rows": f"{result.get('count', 0):,}"})
            
            pbar.close()
    
    batch_cleanup_temp_files()
    logger.info("All files processed, batch cleanup completed.")
    return shard_dir_nas # Return path for next step

if __name__ == "__main__":
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--zst_root_dir", required=True)
    parser.add_argument("--master_index", required=True)
    args = parser.parse_args()
    
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass 

    extract_user_history(args.zst_root_dir, args.master_index)