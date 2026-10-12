import os
import sys
import gc
import shutil
import argparse
import pandas as pd
from datetime import datetime

# Import Modules
try:
    from config import PROCESSING_CONFIG
    from utils import setup_logger
    
    # Import Phases
    from phase1_finder import build_master_index
    from phase2_extractor import extract_user_history
    from post_processing_0302 import (
        extract_features_disk_backed, 
        filter_bots_scientific, 
        assign_shadow_t0, 
        run_psm_matching, 
        build_final_dataset
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Missing module. {e}")
    sys.exit(1)

# Initialize Logger
logger = setup_logger()

# ================= 新增：中间结果保存配置 =================
def get_intermediate_path(step_name):
    """生成标准化的中间结果路径"""
    intermediate_dir = os.path.join(PROCESSING_CONFIG["output_dir"], "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    # 按步骤命名，便于识别
    filename = f"step_{step_name}.parquet"
    return os.path.join(intermediate_dir, filename)

def check_path_exists(path, step_name):
    if os.path.exists(path):
        logger.info(f"✅ {step_name} detected at: {path}")
        return True
    return False

def save_intermediate(df, step_name):
    """保存中间结果，支持压缩和索引"""
    path = get_intermediate_path(step_name)
    # 使用snappy压缩，节省空间且读取快
    df.to_parquet(path, compression="snappy", index=False)
    logger.info(f"💾 Saved intermediate result for {step_name} to: {path}")

def load_intermediate(step_name):
    """加载中间结果"""
    path = get_intermediate_path(step_name)
    logger.info(f"📥 Loading intermediate result for {step_name} from: {path}")
    return pd.read_parquet(path)

def run_full_pipeline(zst_root_dir, skip_phase1=False):
    """
    Executes the full longitudinal phenotyping pipeline.
    
    Flow:
    1. Phase 1: Scan ZST files to identify Patients & Candidate Controls.
    2. Phase 2: Extract full posting history (Optimized Regex Cleaning).
    3. Phase 3: Feature Extraction -> Bot Filter -> Shadow T0 -> PSM -> Final Redaction.
    """
    
    output_dir = PROCESSING_CONFIG["output_dir"]
    master_index_path = os.path.join(output_dir, PROCESSING_CONFIG["master_index_file"])
    final_cohort_path = os.path.join(output_dir, PROCESSING_CONFIG["final_cohort_file"])
    
    logger.info("="*60)
    logger.info(f"🚀 STARTING PIPELINE: {PROCESSING_CONFIG['output_dir']}")
    logger.info(f"   Input Data: {zst_root_dir}")
    logger.info(f"   Cores: {PROCESSING_CONFIG['cpu_cores']} | Batch: {PROCESSING_CONFIG['batch_size']}")
    logger.info("="*60)

    # ================= PHASE 1: PATIENT FINDER =================
    logger.info("\n--- [PHASE 1] Cohort Identification ---")
    
    if check_path_exists(master_index_path, "Master Index"): #skip_phase1 and 
        logger.info("Skipping Phase 1 as requested (Index exists).")
    else:
        # Run Phase 1
        build_master_index(zst_root_dir)
        
    # Validation
    if not os.path.exists(master_index_path):
        logger.critical("❌ Phase 1 failed: Master Index not found.")
        sys.exit(1)
        
    # Free memory
    gc.collect()

    # ================= PHASE 2: HISTORY EXTRACTION =================
    logger.info("\n--- [PHASE 2] Multi-Modal History Extraction (Optimized) ---")
    logger.info("Note: Using Fast Regex Cleaning. PII Redaction is deferred to Phase 3.")
    
    # extract_user_history handles its own resume logic via 'processing_manifest.jsonl'
    # It returns the path to the directory containing the shards (temp_shards)
    
    shard_dir = extract_user_history(zst_root_dir, master_index_path)
    
    if not shard_dir or not os.path.exists(shard_dir):
        logger.critical("❌ Phase 2 failed: Shard directory missing.")
        sys.exit(1)
        
    # Check if shards exist
    num_shards = len([name for name in os.listdir(shard_dir) if name.endswith('.parquet')])
    if num_shards == 0:
        logger.warning("⚠️ Phase 2 completed but no Parquet shards were found. Check inputs.")
    else:
        logger.info(f"✅ Phase 2 extracted {num_shards} shards.")

    # Explicit GC before the heavy math of Phase 3
    gc.collect()

    # ================= PHASE 3: POST-PROCESSING & MATCHING =================
    logger.info("\n--- [PHASE 3] Scientific Rigor & Dataset Construction ---")
    
    try:
        # 3.1 Feature Extraction (Includes Heavy NLP check on samples)
        logger.info("--> 3.1 Extracting Features & Proxies...")
        features_path = get_intermediate_path("3.1_features_extracted")
        if check_path_exists(features_path, "3.1 Features Extracted"):
            features_df = load_intermediate("3.1_features_extracted")
        else:
            # 【关键修改】禁用extract_features_disk_backed中temp_dir的自动删除
            # 方式1：修改函数参数（推荐）
            features_df = extract_features_disk_backed(shard_dir)
            # 方式2：如果函数无参数，临时修改配置
            # PROCESSING_CONFIG["keep_temp_stats"] = True
            
            if features_df.empty:
                logger.critical("❌ No features extracted. Pipeline stopping.")
                sys.exit(1)
            # 保存特征提取结果
            save_intermediate(features_df, "3.1_features_extracted")
            
        # 3.2 Bot Filtering
        logger.info("--> 3.2 Filtering Bots (CONSORT Standards)...")
        clean_features_path = get_intermediate_path("3.2_bot_filtered")
        if check_path_exists(clean_features_path, "3.2 Bot Filtered"):
            clean_features_df = load_intermediate("3.2_bot_filtered")
        else:
            clean_features_df = filter_bots_scientific(features_df)
            # 保存Bot过滤结果
            save_intermediate(clean_features_df, "3.2_bot_filtered")
        
        # Cleanup large DF
        del features_df
        gc.collect()
        
        # 3.3 Shadow T0 Assignment
        logger.info("--> 3.3 Assigning Shadow T0s (Incidence Density Sampling)...")
        t0_assigned_path = get_intermediate_path("3.3_shadow_t0_assigned")
        if check_path_exists(t0_assigned_path, "3.3 Shadow T0 Assigned"):
            t0_assigned_df = load_intermediate("3.3_shadow_t0_assigned")
        else:
            t0_assigned_df = assign_shadow_t0(clean_features_df)
            # 保存Shadow T0分配结果
            save_intermediate(t0_assigned_df, "3.3_shadow_t0_assigned")
        
        del clean_features_df
        gc.collect()
        
        # 3.4 Propensity Score Matching
        logger.info("--> 3.4 Running Propensity Score Matching (PSM)...")
        matched_meta_path = get_intermediate_path("3.4_psm_matched")
        if check_path_exists(matched_meta_path, "3.4 PSM Matched"):
            matched_meta_df = load_intermediate("3.4_psm_matched")
        else:
            matched_meta_df = run_psm_matching(t0_assigned_df)
            
            if matched_meta_df.empty:
                logger.critical("❌ Matching resulted in empty cohort. Adjust PSM config.")
                sys.exit(1)
            # 保存PSM匹配结果
            save_intermediate(matched_meta_df, "3.4_psm_matched")
        
        del t0_assigned_df
        gc.collect()
        
        # 3.5 Final Dataset Build (The Final Privacy Firewall)
        logger.info("--> 3.5 Building Final Redacted Dataset...")
        # 最终数据集已有指定路径，无需保存到中间目录
        build_final_dataset(matched_meta_df, shard_dir, final_cohort_path)
        
        logger.info("✅ Phase 3 Complete.")

    except Exception as e:
        logger.critical(f"❌ Phase 3 Crashed: {e}", exc_info=True)
        sys.exit(1)

    # ================= COMPLETION =================
    logger.info("\n" + "="*60)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Final Output: {final_cohort_path.replace('.parquet', '_dataset')}")
    logger.info(f"Intermediate Results: {os.path.join(PROCESSING_CONFIG['output_dir'], 'intermediate_results')}")
    logger.info(f"Temp Stats Directory: {os.path.join(PROCESSING_CONFIG['WORK_DIR'], 'temp_stats_agg')}")
    logger.info("="*60)
    
    # Recommendation on Temp Files
    logger.info(f"\n[NOTE] Intermediate shards with RAW PII are located at:\n  -> {shard_dir}")
    logger.info("These allow for re-matching without re-extraction.")
    logger.info("If storage is tight or for security compliance, delete them manually:")
    logger.info(f"  rm -rf {shard_dir}")
    # 新增：提示中间结果和临时统计目录的删除方式
    logger.info(f"\n[NOTE] To delete intermediate results (save space):")
    logger.info(f"  rm -rf {os.path.join(PROCESSING_CONFIG['output_dir'], 'intermediate_results')}")
    logger.info(f"[NOTE] To delete temp stats directory:")
    logger.info(f"  rm -rf {os.path.join(PROCESSING_CONFIG['WORK_DIR'], 'temp_stats_agg')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Longitudinal Cohort Pipeline")
    
    parser.add_argument(
        "--zst_root_dir", 
        default="//mnt/path/to/reddit",
        help="Root directory containing 'comments/' and 'submissions/' subfolders with ZST files"
    )
    
    parser.add_argument(
        "--skip_phase1", 
        action="store_true",
        help="Skip Patient Finder if master_index.parquet already exists."
    )
    
    # 新增参数：是否跳过已保存的中间步骤（断点续跑）
    parser.add_argument(
        "--resume_from", 
        default=None,
        help="Resume pipeline from a specific step (e.g., 3.1, 3.2, 3.3, 3.4)"
    )
    
    args = parser.parse_args()
    
    # Verify input dir
    if not os.path.isdir(args.zst_root_dir):
        print(f"Error: Input directory not found: {args.zst_root_dir}")
        sys.exit(1)
        
    run_full_pipeline(args.zst_root_dir, skip_phase1=args.skip_phase1)
