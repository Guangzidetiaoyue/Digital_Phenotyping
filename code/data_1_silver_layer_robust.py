import os
import glob
import json
import shutil
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime
import gc
import traceback
import psutil
import random

random.seed(42)

# ================= CONFIGURATION =================
INPUT_ROOT = "/path/to/reddit/emotion_disorder_1225/cohort_v1/final_longitudinal_data_dataset"
OUTPUT_ROOT = "/path/to/reddit/emotion_disorder_1225/cohort_v1/silver_v2"
REGISTRY_DIR = os.path.join(OUTPUT_ROOT, "_registry")
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")
TEMP_DIR = os.path.join("/tmp", "reddit_silver_temp")
NAS_TEMP_DIR = os.path.join(OUTPUT_ROOT, "_temp_work")

NUM_PARTITIONS = 500
BATCH_SIZE = 100000
# 内存安全阈值：300GB机器，留50GB余量给系统和写入峰值
MAX_MEM_GB = 240
# 单文件内存上限估算：超过此值时逐batch flush
FLUSH_MEM_GB = 200

PROGRESS_FILE = os.path.join(REGISTRY_DIR, "progress.json")
FAILED_FILE = os.path.join(REGISTRY_DIR, "failed_files.json")
ERROR_LOG_FILE = os.path.join(REGISTRY_DIR, "corrupt_files.log")
STATS_FILE = os.path.join(REGISTRY_DIR, "processing_stats.json")

# ================= STRICT SCHEMA =================
SILVER_SCHEMA = pa.schema([
    ("user_id_hash", pa.string()),
    ("post_timestamp", pa.int64()),
    ("post_datetime", pa.string()),
    ("relative_days", pa.float64()),
    ("data_type", pa.string()),
    ("interaction_type", pa.string()),
    ("parent_id", pa.string()),
    ("link_id", pa.string()),
    ("is_top_level", pa.bool_()),
    ("num_comments", pa.int64()),
    ("subreddit", pa.string()),
    ("score", pa.float64()),
    ("controversiality", pa.int64()),
    ("gilded", pa.int64()),
    ("is_edited", pa.bool_()),
    ("is_nsfw", pa.bool_()),
    ("has_media", pa.bool_()),
    ("domain", pa.string()),
    ("author_flair", pa.string()),
    ("clean_text", pa.string()),
    ("flag_suicide", pa.bool_()),
    ("flag_self_harm", pa.bool_()),
    ("group", pa.string()),
    ("disorder_group", pa.string()),
    ("t0_date", pa.string()),
    ("is_diagnosis_post", pa.bool_()),
    ("is_retrospective", pa.bool_()),
    ("post_id", pa.string()),
    ("assigned_t0", pa.int64()),
    ("__index_level_0__", pa.int64())
])

# Schema中不包含partition_id的写出用schema
WRITE_SCHEMA = SILVER_SCHEMA

for dir_path in [REGISTRY_DIR, DATA_DIR, TEMP_DIR, NAS_TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(REGISTRY_DIR, "processing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ================= UTILITY FUNCTIONS =================

def copy_nas_file_to_local(nas_file_path, local_temp_dir):
    """将NAS文件拷贝到本地临时目录，返回本地路径或None"""
    try:
        file_name = os.path.basename(nas_file_path)
        local_file_path = os.path.join(
            local_temp_dir, f"{datetime.now().timestamp()}_{file_name}"
        )
        shutil.copy2(nas_file_path, local_file_path)

        nas_size = os.path.getsize(nas_file_path)
        local_size = os.path.getsize(local_file_path)
        if nas_size != local_size:
            logger.error(
                f"拷贝文件大小不匹配: {nas_file_path} "
                f"(NAS:{nas_size} vs 本地:{local_size})"
            )
            os.remove(local_file_path)
            return None

        logger.info(f"文件已拷贝到本地: {local_file_path} ({nas_size / 1024 / 1024:.1f}MB)")
        return local_file_path
    except Exception as e:
        logger.error(f"拷贝NAS文件失败: {nas_file_path}, 错误: {str(e)}")
        return None


def safe_remove_local_file(local_file_path):
    """安全删除本地临时文件"""
    try:
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
            logger.debug(f"本地临时文件已删除: {local_file_path}")
    except Exception as e:
        logger.warning(f"删除本地临时文件失败: {local_file_path}, 错误: {str(e)}")


def get_partition_index(user_hash):
    try:
        return int(user_hash[:5], 16) % NUM_PARTITIONS
    except (ValueError, TypeError, AttributeError):
        return 0


def load_json_set(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
    return set()


def save_json_set(data_set, filepath):
    try:
        with open(filepath, "w") as f:
            json.dump(list(data_set), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")


def log_corrupted_file(filepath, error_msg):
    try:
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(
                f"[{datetime.now().isoformat()}] CORRUPT FILE: {filepath}\n"
                f"ERROR: {error_msg}\n\n"
            )
    except Exception as e:
        logger.error(f"Failed to log corrupted file: {e}")


def get_mem_used_gb():
    """获取当前系统已使用内存（GB）"""
    return psutil.virtual_memory().used / (1024 ** 3)


def get_process_mem_gb():
    """获取当前进程RSS内存（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def cast_batch_to_schema(batch):
    """
    将一个RecordBatch强制转换为SILVER_SCHEMA。
    先尝试直接cast，失败则走pandas逐字段转换。
    返回转换后的RecordBatch。
    """
    try:
        return batch.cast(SILVER_SCHEMA, safe=False)
    except Exception:
        pass

    # Fallback: 通过pandas逐字段转换
    df = batch.to_pandas()
    for field in SILVER_SCHEMA:
        field_name = field.name
        if field_name not in df.columns:
            if pa.types.is_boolean(field.type):
                df[field_name] = False
            elif pa.types.is_integer(field.type):
                df[field_name] = 0
            elif pa.types.is_floating(field.type):
                df[field_name] = 0.0
            elif pa.types.is_string(field.type):
                df[field_name] = ""
            continue
        try:
            if pa.types.is_boolean(field.type):
                df[field_name] = df[field_name].astype(bool)
            elif pa.types.is_integer(field.type):
                df[field_name] = (
                    pd.to_numeric(df[field_name], errors='coerce')
                    .fillna(0)
                    .astype('int64')
                )
            elif pa.types.is_floating(field.type):
                df[field_name] = (
                    pd.to_numeric(df[field_name], errors='coerce')
                    .astype('float64')
                )
            elif pa.types.is_string(field.type):
                df[field_name] = df[field_name].astype(str)
        except Exception as e:
            logger.error(f"字段 {field_name} 类型转换失败: {e}")
            raise

    table = pa.Table.from_pandas(df, schema=SILVER_SCHEMA)
    result_batch = table.to_batches()[0]
    del df, table
    return result_batch


def flush_partition_buffers(partition_buffers, flush_tag):
    """
    将partition_buffers中的数据写出到磁盘（增量追加方式）。
    每个分区写一个小文件，文件名包含flush_tag以保证唯一性。
    写完后清空buffers并释放内存。
    返回写出的分区数。
    """
    written = 0
    for pid in range(NUM_PARTITIONS):
        df_list = partition_buffers.get(pid, [])
        if not df_list:
            continue

        non_empty = [df for df in df_list if not df.empty]
        if not non_empty:
            partition_buffers[pid] = []
            continue

        try:
            combined_df = pd.concat(non_empty, ignore_index=True, sort=False)
            # 确保写出时不包含partition_id列
            if "partition_id" in combined_df.columns:
                combined_df = combined_df.drop(columns=["partition_id"])

            table = pa.Table.from_pandas(combined_df, schema=WRITE_SCHEMA)

            part_dir = os.path.join(DATA_DIR, f"part_{pid:04d}")
            os.makedirs(part_dir, exist_ok=True)

            out_file = os.path.join(part_dir, f"chunk_{flush_tag}.parquet")
            pq.write_table(
                table,
                out_file,
                compression='zstd',
                use_dictionary=True,
                data_page_size=1024 * 1024,
                row_group_size=100000
            )
            written += 1
            del combined_df, table
        except Exception as e:
            logger.error(f"Failed to write partition {pid:04d}: {e}")

        # 清空该分区buffer
        partition_buffers[pid] = []

    gc.collect()
    return written


def flush_registry_buffer(registry_buffer, flush_tag):
    """将registry_buffer写出为一个临时parquet文件，清空buffer"""
    if not registry_buffer:
        return
    non_empty = [df for df in registry_buffer if not df.empty]
    if not non_empty:
        registry_buffer.clear()
        return
    try:
        reg_df = pd.concat(non_empty).drop_duplicates()
        reg_out = os.path.join(NAS_TEMP_DIR, f"reg_chunk_{flush_tag}.parquet")
        reg_df.to_parquet(reg_out, index=False, compression="zstd")
        del reg_df
    except Exception as e:
        logger.error(f"Failed to save registry chunk: {e}")
    registry_buffer.clear()
    gc.collect()


def merge_registry_batch(reg_files, output_path):
    """分批合并注册表文件，避免一次性加载所有注册表OOM"""
    if not reg_files:
        logger.warning("No registry files to merge!")
        return

    final_registry = pd.DataFrame(columns=["user_id_hash", "partition_id"])
    merge_batch_size = 20

    total_batches = (len(reg_files) + merge_batch_size - 1) // merge_batch_size
    for i in range(0, len(reg_files), merge_batch_size):
        batch_idx = i // merge_batch_size + 1
        logger.info(f"Merging registry batch {batch_idx}/{total_batches}")

        chunk_batch = reg_files[i:i + merge_batch_size]
        try:
            dfs = []
            for f in chunk_batch:
                try:
                    dfs.append(pd.read_parquet(f))
                except Exception as e:
                    logger.error(f"Failed to read registry file {f}: {e}")
            if dfs:
                df_batch = pd.concat(dfs)
                final_registry = pd.concat(
                    [final_registry, df_batch]
                ).drop_duplicates("user_id_hash", keep="last")
                del df_batch, dfs
                gc.collect()
        except Exception as e:
            logger.error(f"Failed to merge registry batch {batch_idx}: {e}")
            continue

    final_registry["processed_at"] = datetime.now().isoformat()
    final_registry.to_parquet(output_path, index=False, compression="zstd")
    logger.info(f"Registry merged: {len(final_registry)} unique users")
    del final_registry
    gc.collect()


# ================= MAIN PROCESSING =================

def build_silver_layer():
    all_files = sorted(
        glob.glob(os.path.join(INPUT_ROOT, "**/*.parquet"), recursive=True)
    )
    random.shuffle(all_files)
    if not all_files:
        logger.error("No input parquet files found!")
        return

    # Load Progress and Failed Files
    processed_files = load_json_set(PROGRESS_FILE)
    failed_files = load_json_set(FAILED_FILE)

    # 正常模式：跳过已成功处理的文件，重试之前失败的文件
    pending_files = [f for f in all_files if f not in processed_files]

    stats = {
        "total_files": len(all_files),
        "processed": len(processed_files),
        "permanently_failed": len(failed_files),
        "pending": len(pending_files),
        "success_this_run": 0,
        "failed_this_run": 0,
        "total_rows": 0,
        "start_time": datetime.now().isoformat()
    }

    logger.info("=== Processing Stats ===")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Already processed: {stats['processed']}")
    logger.info(f"Previously failed (will retry): {stats['permanently_failed']}")
    logger.info(f"Pending: {stats['pending']}")
    logger.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    logger.info(f"Memory safety threshold: {MAX_MEM_GB}GB")

    # ====================================================
    # 核心变化：逐文件处理 + 内存驱动的flush策略
    # 不再攒20个文件才写盘，而是：
    #   - 每个文件处理完后检查内存
    #   - 内存超过阈值时flush到磁盘
    #   - 每处理N个文件强制flush一次（防止小文件积累过多）
    # ====================================================

    partition_buffers = {pid: [] for pid in range(NUM_PARTITIONS)}
    registry_buffer = []
    flush_counter = 0  # flush编号，用于生成唯一文件名
    files_since_flush = 0
    FORCE_FLUSH_EVERY_N_FILES = 10  # 每处理10个文件强制flush

    for file_idx, file_path in enumerate(
        tqdm(pending_files, desc="Processing Files")
    ):
        local_file = None
        try:
            # --- 内存检查（处理文件前）---
            mem_used = get_mem_used_gb()
            proc_mem = get_process_mem_gb()
            if mem_used > MAX_MEM_GB:
                logger.warning(
                    f"Memory high before file {file_idx}: "
                    f"system={mem_used:.1f}GB, process={proc_mem:.1f}GB. "
                    f"Flushing buffers to disk..."
                )
                flush_counter += 1
                flush_tag = f"{flush_counter}_{int(datetime.now().timestamp())}"
                n_written = flush_partition_buffers(partition_buffers, flush_tag)
                flush_registry_buffer(registry_buffer, flush_tag)
                files_since_flush = 0
                logger.info(f"Emergency flush complete: {n_written} partitions written")

                # 再检查一次，如果还高则等待GC
                gc.collect()
                mem_used = get_mem_used_gb()
                if mem_used > MAX_MEM_GB:
                    logger.error(
                        f"Memory still high after flush: {mem_used:.1f}GB. "
                        f"Continuing cautiously..."
                    )

            # --- 1. 拷贝文件到本地 ---
            local_file = copy_nas_file_to_local(file_path, TEMP_DIR)
            if not local_file:
                raise Exception("NAS文件拷贝到本地失败")

            logger.info(
                f"[{file_idx+1}/{len(pending_files)}] 开始处理: "
                f"{os.path.basename(file_path)}"
            )

            # --- 2. 打开文件并迭代处理 ---
            pf = pq.ParquetFile(local_file)
            file_rows = 0
            logger.info(
                f"文件 {os.path.basename(file_path)} 共 {pf.num_row_groups} 个 row group"
            )

            for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
                try:
                    # Schema转换
                    batch = cast_batch_to_schema(batch)

                    # 转为DataFrame
                    df = batch.to_pandas()
                    batch_rows = len(df)
                    file_rows += batch_rows
                    del batch  # 释放arrow batch

                    # 计算分区
                    if "user_id_hash" not in df.columns or df["user_id_hash"].isnull().all():
                        logger.warning(
                            f"文件 {os.path.basename(file_path)} batch {batch_idx} "
                            f"无有效 user_id_hash，跳过"
                        )
                        del df
                        continue

                    df["partition_id"] = df["user_id_hash"].apply(get_partition_index)

                    # 更新注册表
                    reg_df = df[["user_id_hash", "partition_id"]].drop_duplicates()
                    if not reg_df.empty:
                        registry_buffer.append(reg_df)

                    # 按分区分组并存入buffer
                    for pid, group in df.groupby("partition_id"):
                        partition_buffers[pid].append(
                            group.drop(columns=["partition_id"])
                        )

                    del df
                    gc.collect()

                    # --- 每个batch后检查内存 ---
                    mem_used = get_mem_used_gb()
                    if mem_used > FLUSH_MEM_GB:
                        logger.warning(
                            f"Memory {mem_used:.1f}GB > {FLUSH_MEM_GB}GB "
                            f"during batch processing. Mid-file flush..."
                        )
                        flush_counter += 1
                        flush_tag = (
                            f"{flush_counter}_{int(datetime.now().timestamp())}"
                        )
                        flush_partition_buffers(partition_buffers, flush_tag)
                        flush_registry_buffer(registry_buffer, flush_tag)
                        files_since_flush = 0
                        gc.collect()

                except Exception as batch_e:
                    logger.error(
                        f"文件 {os.path.basename(file_path)} "
                        f"batch {batch_idx} 处理失败: {batch_e}"
                    )
                    # 跳过损坏的batch，继续处理下一个batch
                    continue

            # --- 3. 文件处理成功 ---
            # 释放ParquetFile引用
            del pf

            # 先删除本地文件释放/tmp空间
            if local_file:
                safe_remove_local_file(local_file)
                local_file = None

            processed_files.add(file_path)
            # 如果之前在失败列表中，移除
            if file_path in failed_files:
                failed_files.discard(file_path)

            stats["success_this_run"] += 1
            stats["total_rows"] += file_rows
            files_since_flush += 1

            logger.info(
                f"文件 {os.path.basename(file_path)} 处理完成, "
                f"行数: {file_rows}, "
                f"进程内存: {get_process_mem_gb():.1f}GB"
            )

            # --- 4. 定期flush（每N个文件或内存较高时）---
            mem_used = get_mem_used_gb()
            if (files_since_flush >= FORCE_FLUSH_EVERY_N_FILES
                    or mem_used > FLUSH_MEM_GB):
                logger.info(
                    f"Periodic flush: {files_since_flush} files since last flush, "
                    f"memory: {mem_used:.1f}GB"
                )
                flush_counter += 1
                flush_tag = f"{flush_counter}_{int(datetime.now().timestamp())}"
                n_written = flush_partition_buffers(partition_buffers, flush_tag)
                flush_registry_buffer(registry_buffer, flush_tag)
                files_since_flush = 0
                logger.info(f"Flush complete: {n_written} partitions written")
                gc.collect()

            # --- 5. 定期保存进度（每10个文件）---
            if (file_idx + 1) % 10 == 0:
                save_json_set(processed_files, PROGRESS_FILE)
                save_json_set(failed_files, FAILED_FILE)
                stats["processed"] = len(processed_files)
                stats["permanently_failed"] = len(failed_files)
                logger.info(
                    f"Progress saved: {len(processed_files)}/{len(all_files)} "
                    f"processed, {len(failed_files)} failed"
                )

        except Exception as e:
            logger.error(
                f"Failed to process {os.path.basename(file_path)}: {e}"
            )
            log_corrupted_file(file_path, traceback.format_exc())
            failed_files.add(file_path)
            stats["failed_this_run"] += 1
        finally:
            if local_file:
                safe_remove_local_file(local_file)

    # ====================================================
    # 处理完所有文件后，最终flush剩余buffer
    # ====================================================
    logger.info("All files iterated. Final flush of remaining buffers...")
    flush_counter += 1
    flush_tag = f"{flush_counter}_{int(datetime.now().timestamp())}_final"
    n_written = flush_partition_buffers(partition_buffers, flush_tag)
    flush_registry_buffer(registry_buffer, flush_tag)
    logger.info(f"Final flush complete: {n_written} partitions written")

    # 最终保存进度
    save_json_set(processed_files, PROGRESS_FILE)
    save_json_set(failed_files, FAILED_FILE)

    del partition_buffers, registry_buffer
    gc.collect()

    # ================= FINALIZATION =================
    logger.info("All files processed. Merging Global User Registry...")
    reg_files = glob.glob(os.path.join(NAS_TEMP_DIR, "reg_chunk_*.parquet"))
    if reg_files:
        merge_registry_batch(
            reg_files, os.path.join(REGISTRY_DIR, "user_lookup.parquet")
        )
    else:
        logger.warning("No registry chunk files found to merge.")

    stats["end_time"] = datetime.now().isoformat()
    stats["processed"] = len(processed_files)
    stats["permanently_failed"] = len(failed_files)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    # 清理临时目录
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    shutil.rmtree(NAS_TEMP_DIR, ignore_errors=True)

    logger.info("=== Silver Layer Build Completed ===")
    logger.info(f"Success this run: {stats['success_this_run']}")
    logger.info(f"Failed this run: {stats['failed_this_run']}")
    logger.info(f"Total rows processed: {stats['total_rows']}")
    logger.info(f"Total processed files: {stats['processed']}")
    logger.info(f"Total failed files: {stats['permanently_failed']}")
    logger.info(f"Corrupt files log: {ERROR_LOG_FILE}")
    logger.info(f"Processing stats: {STATS_FILE}")


if __name__ == "__main__":
    try:
        build_silver_layer()
    except Exception as e:
        logger.error(f"Fatal error in build_silver_layer: {e}", exc_info=True)
    finally:
        # 确保程序异常退出时也清理临时文件
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        # 注意：NAS_TEMP_DIR中的registry chunk在异常退出时保留，
        # 以便下次运行时可以继续合并
        logger.info("Cleanup complete.")