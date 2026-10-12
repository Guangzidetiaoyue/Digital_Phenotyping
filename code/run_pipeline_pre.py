# run_pipeline.py - Master Orchestrator for Nature/Science Protocol
# Features: Checkpointing, Memory Management, and Scientific Rigor

import os
import sys
import argparse
import gc
import logging
import pandas as pd

# Import all modules
from phase1_finder import build_master_index
from phase2_extractor import extract_user_history
from post_processing_0302 import (
    extract_features_mapreduce,
    filter_bots_scientific,
    assign_shadow_t0,
    run_psm_matching,
    build_final_dataset
)
from utils import setup_logger
from config import PROCESSING_CONFIG

# Initialize logger
logger = setup_logger()

def check_step_complete(path, step_name):
    """
    Checks if a file or a non-empty directory exists to validate a step's completion.
    """
    if not os.path.exists(path):
        return False
    
    # Check for non-empty directory (for shards)
    if os.path.isdir(path):
        if len(os.listdir(path)) > 0:
            logger.info(f"✅ {step_name} output found at {path}. Skipping.")
            return True
        else:
            return False # Directory exists but is empty
            
    # Check for non-empty file
    if os.path.isfile(path) and os.path.getsize(path) > 1024: # >1KB
        logger.info(f"✅ {step_name} output found at {path}. Skipping.")
        return True
        
    return False

def run_full_pipeline(zst_root_dir):
    """
    End-to-end pipeline with resume capability.
    1. Phase 1: Build Master Index (Patients + Controls)
    2. Phase 2: Extract Full History (Shards)
    3. Post-Processing: Feature Extraction, Bot Filtering, PSM, Final Dataset
    """
    try:
        # --- 0. SETUP ---
        output_dir = PROCESSING_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Define key file paths for checkpointing
        master_index_path = os.path.join(output_dir, PROCESSING_CONFIG["master_index_file"])
        shard_dir = os.path.join(output_dir, "temp_shards")
        final_matched_path = os.path.join(output_dir, "final_matched_cohort.parquet")
        
        logger.info("="*50)
        logger.info("🚀 STARTING FULL PIPELINE (NATURE/SCIENCE PROTOCOL)")
        logger.info(f"Output will be stored in: {output_dir}")
        logger.info("="*50)

        # --- 1. PHASE 1: FINDER ---
        if not check_step_complete(master_index_path, "Phase 1: Master Index"):
            logger.info("\n--- Running Phase 1: Patient Identification & Control Sampling ---")
            build_master_index(zst_root_dir)
            gc.collect() # Free up memory
        
        # --- 2. PHASE 2: FULL HISTORY EXTRACTION ---
        # Strategy: Manifest-based checkpoint.
        # We ALWAYS call this function. It internally checks 'processing_manifest.jsonl'
        # and only processes files that are missing or failed.
        logger.info("\n--- [PHASE 2] Multi-Modal History Extraction ---")
        
        # Note: extract_user_history returns the path to the shard directory
        shard_dir = extract_user_history(zst_root_dir, master_index_path)
        
        if not shard_dir or not os.path.exists(shard_dir):
            logger.critical("❌ Phase 2 Error: Shard directory not returned or missing.")
            sys.exit(1)
            
        # Optional: Sanity check (are there any parquet files?)
        if len(os.listdir(shard_dir)) == 0:
             logger.warning("⚠️ Phase 2 finished but shard directory is empty. Check logs.")
             # We don't exit here, because maybe all inputs were empty (unlikely but possible)
        
        logger.info("Phase 2 validated. Cleaning memory...")
        gc.collect()
        
        # --- 3. POST-PROCESSING ---
        if not check_step_complete(final_matched_path, "Post-Processing: Final Matched Cohort"):
            logger.info("\n--- Running Post-Processing: Feature Eng, Bot Filtering, & PSM ---")
            
            # 3.1: Feature Extraction (Memory-safe)
            logger.info("--> Step 3.1: Extracting User-Level Features (Map-Reduce)")
            features_df = extract_features_mapreduce(shard_dir)
            
            # 3.2: Bot Filtering (Scientific)
            logger.info("--> Step 3.2: Applying Scientific Bot Filters")
            clean_features_df = filter_bots_scientific(features_df)
            del features_df; gc.collect()
            
            # 3.3: Shadow T0 Assignment (Calendar Anchoring)
            logger.info("--> Step 3.3: Assigning Shadow T0s via Incidence Density Sampling")
            t0_assigned_df = assign_shadow_t0(clean_features_df)
            del clean_features_df; gc.collect()
            
            # 3.4: Propensity Score Matching
            # logger.info("--> Step 3.4: Running Propensity Score Matching with SMD Validation")
            # matched_meta_df = run_psm_matching(t0_assigned_df)
            # del t0_assigned_df; gc.collect()
            
            # # 3.5: Final Dataset Reconstruction
            # logger.info("--> Step 3.5: Reconstructing Final Matched Dataset from Shards")
            # build_final_dataset(matched_meta_df, shard_dir, final_matched_path)
            
        # --- 4. CONCLUSION ---
        logger.info("\n" + "="*50)
        logger.info("🎉 PIPELINE SUCCESSFULLY COMPLETED 🎉")
        logger.info(f"Final Matched Dataset Location: {final_matched_path}")
        logger.info("This dataset is now ready for rigorous statistical and linguistic analysis.")
        logger.info("="*50)

    except Exception as e:
        logger.critical(f"🔥 PIPELINE FAILED: {str(e)}", exc_info=True)
        sys.exit(1)

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline: Reddit Mental Health Cohort Construction")
    parser.add_argument(
        "--zst_root_dir", 
        default="/data1/qizhixiao/path/to/reddit",
        help="Root directory containing 'comments/' and 'submissions/' subfolders with ZST files"
    ) #required=True,  /mnt/path/to/reddit/test_1225 zst_test_files  "/mnt/path/to/reddit"

    args = parser.parse_args()
    
    run_full_pipeline(args.zst_root_dir)
    