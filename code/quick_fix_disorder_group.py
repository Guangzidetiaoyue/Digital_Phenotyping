"""
quick_fix_disorder_group_v2.py
补全 PSM matched 文件和 user_meta 中的 disorder_group 列。

v2 修复:
  1. 优先从 s01 特征文件获取（极快、最完整）
  2. 补充从 Silver 分区扫描
  3. 修复 user_meta 路径（.parquet 后缀问题）
  4. 处理已有 disorder_group 但含 unknown 的情况
  5. fallback 扫描去除 [:20] 限制
"""

import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ======================== 配置 ========================
SILVER_DATA_DIR = "/path/to/project/cohort_v1/silver_v2"
PSM_MATCHED_PATH = "/path/to/project/cohort_v1/intermediate_results/step_3.4_psm_matched.parquet"

# ✅ 修复1: user_meta 的实际路径
# post_processing.py 中: meta_path = out_file + '_user_meta.parquet'
# out_file 以 .parquet 结尾，所以实际文件名多了一个 .parquet
USER_META_PATH_CANDIDATES = [
    "/path/to/project/cohort_v1/intermediate_results/final_longitudinal_data.parquet",  # 原脚本路径 
    #"/path/to/project/cohort_v1/final_longitudinal_data_user_meta.parquet"
]

# ✅ 新增: s01 特征文件路径（最快最完整的来源）
S01_FEATURES_CANDIDATES = [
    "/path/to/project/module1_results/features/user_features_raw.parquet",
    # 尝试其他可能路径
    os.path.join(os.path.dirname(PSM_MATCHED_PATH), "user_features_raw.parquet"),
    os.path.join(os.path.dirname(PSM_MATCHED_PATH), "step_3.1_features_extracted.parquet"),
]

REGISTRY_PATH = os.path.join(SILVER_DATA_DIR, "_registry", "user_lookup.parquet")
BATCH_SIZE = 100000


# ======================== 多源 disorder_group 加载 ========================

def load_disorder_map_from_s01(target_uids: set) -> dict:
    """
    ✅ 最优源: 从 s01 特征文件获取 disorder_group。
    
    优势:
      - 1个文件 vs 500个分区，读取极快
      - s01 已从 Silver 数据提取了完整 disorder_group
      - 覆盖 ~127,000 患者
    
    返回: {user_id_hash: disorder_group}
    """
    for feat_path in S01_FEATURES_CANDIDATES:
        if not os.path.exists(feat_path):
            continue
        
        print(f"  Trying s01 features: {feat_path}")
        try:
            # 只读需要的列，节省内存
            available_cols = pq.ParquetFile(feat_path).schema.names
            read_cols = [c for c in ["user_id_hash", "group", "disorder_group"] if c in available_cols]
            
            if "disorder_group" not in available_cols:
                print(f"    ✗ No disorder_group column in {feat_path}")
                continue
            
            df = pd.read_parquet(feat_path, columns=read_cols)
            
            # 过滤目标用户 + 有效 disorder_group
            valid_mask = (
                df["user_id_hash"].isin(target_uids) &
                df["disorder_group"].notna()
            )
            valid = df[valid_mask].copy()
            
            # 排除无效标签
            invalid_labels = {"control", "control_candidate", "unknown", "nan", "None", "", "NaT"}
            valid = valid[~valid["disorder_group"].astype(str).isin(invalid_labels)]
            
            # 去重（每个用户取第一条）
            valid = valid.drop_duplicates(subset=["user_id_hash"])
            
            disorder_map = dict(zip(valid["user_id_hash"], valid["disorder_group"].astype(str)))
            
            print(f"    ✓ Recovered from s01: {len(disorder_map)}/{len(target_uids)} patients")
            
            del df, valid
            return disorder_map
            
        except Exception as e:
            print(f"    ✗ Error reading {feat_path}: {e}")
            continue
    
    print("  ✗ No s01 features file found")
    return {}


def load_disorder_map_from_silver(target_uids: set) -> dict:
    """从 Silver 数据扫描 disorder_group（补充源）"""
    disorder_map = {}
    
    # 策略1: 用注册表定向扫描
    if os.path.exists(REGISTRY_PATH):
        print(f"  Using registry for targeted scan: {REGISTRY_PATH}")
        try:
            reg = pd.read_parquet(REGISTRY_PATH, columns=["user_id_hash", "partition_id"])
            reg = reg[reg["user_id_hash"].isin(target_uids)]
            
            if len(reg) == 0:
                print("    Registry has no matching users for remaining targets")
                return {}
            
            partition_users = reg.groupby("partition_id")["user_id_hash"].apply(set).to_dict()
            print(f"    Remaining users spread across {len(partition_users)} partitions")
            
            for pid, users in tqdm(partition_users.items(), desc="  Scanning partitions"):
                if not (target_uids - set(disorder_map.keys())):
                    break
                
                part_dir = os.path.join(SILVER_DATA_DIR, "data", f"part_{pid:04d}")
                if not os.path.isdir(part_dir):
                    part_dir = os.path.join(SILVER_DATA_DIR, f"part_{pid:04d}")
                if not os.path.isdir(part_dir):
                    continue
                
                remaining = users - set(disorder_map.keys())
                # ✅ 修复: 去掉 [:20] 限制，扫描所有文件
                for pf_path in glob.glob(os.path.join(part_dir, "*.parquet")):
                    if not remaining:
                        break
                    try:
                        pf = pq.ParquetFile(pf_path)
                        if "disorder_group" not in pf.schema.names:
                            continue
                        for batch in pf.iter_batches(
                            batch_size=BATCH_SIZE,
                            columns=["user_id_hash", "disorder_group"]
                        ):
                            df = batch.to_pandas()
                            mask = (
                                df["user_id_hash"].isin(remaining) & 
                                df["disorder_group"].notna()
                            )
                            matched = df[mask][["user_id_hash", "disorder_group"]].drop_duplicates(
                                subset=["user_id_hash"]
                            )
                            invalid_labels = {"control", "control_candidate", "unknown", "nan", "None", ""}
                            for uid, dg in matched.values:
                                dg_str = str(dg)
                                if dg_str not in invalid_labels:
                                    disorder_map[uid] = dg_str
                                    remaining.discard(uid)
                            del df, matched
                    except Exception as e:
                        print(f"    Error: {pf_path}: {e}")
            
        except Exception as e:
            print(f"  Registry scan failed: {e}")
    
    else:
        # 策略2: 全分区扫描（无 [:20] 限制）
        print("  Registry not found, full partition scan...")
        data_dir = os.path.join(SILVER_DATA_DIR, "data")
        base_dir = data_dir if os.path.isdir(data_dir) else SILVER_DATA_DIR
        
        remaining = set(target_uids)
        for entry in sorted(os.listdir(base_dir)):
            if not remaining:
                break
            full_path = os.path.join(base_dir, entry)
            if not (os.path.isdir(full_path) and entry.startswith("part_")):
                continue
            
            # ✅ 修复: 去掉 [:20]，扫描全部文件
            for pf_path in glob.glob(os.path.join(full_path, "*.parquet")):
                if not remaining:
                    break
                try:
                    pf = pq.ParquetFile(pf_path)
                    if "disorder_group" not in pf.schema.names:
                        continue
                    for batch in pf.iter_batches(
                        batch_size=BATCH_SIZE,
                        columns=["user_id_hash", "disorder_group"]
                    ):
                        df = batch.to_pandas()
                        mask = df["user_id_hash"].isin(remaining) & df["disorder_group"].notna()
                        matched = df[mask][["user_id_hash", "disorder_group"]].drop_duplicates(
                            subset=["user_id_hash"]
                        )
                        invalid_labels = {"control", "control_candidate", "unknown", "nan", "None", ""}
                        for uid, dg in matched.values:
                            dg_str = str(dg)
                            if dg_str not in invalid_labels:
                                disorder_map[uid] = dg_str
                                remaining.discard(uid)
                        del df, matched
                except Exception:
                    pass
    
    return disorder_map


def load_disorder_map_multi_source(target_uids: set) -> dict:
    """
    ✅ 多源组合加载，最大化恢复率。
    
    优先级:
      1. s01 特征文件（极快，覆盖 ~127K 患者）
      2. Silver 分区扫描（较慢，覆盖额外用户）
    """
    disorder_map = {}
    
    # ========== 源1: s01 特征文件 ==========
    print("\n[Source 1] Loading from s01 features file...")
    s01_map = load_disorder_map_from_s01(target_uids)
    disorder_map.update(s01_map)
    
    remaining = target_uids - set(disorder_map.keys())
    print(f"  After s01: {len(disorder_map)} recovered, {len(remaining)} still missing")
    
    # ========== 源2: Silver 分区扫描 ==========
    if remaining:
        print(f"\n[Source 2] Scanning Silver data for {len(remaining)} remaining users...")
        silver_map = load_disorder_map_from_silver(remaining)
        disorder_map.update(silver_map)
        
        remaining = target_uids - set(disorder_map.keys())
        print(f"  After Silver scan: {len(disorder_map)} recovered, {len(remaining)} still missing")
    
    if remaining:
        print(f"\n  ⚠️ {len(remaining)} patients could not be recovered from any source")
        print(f"     These users will have disorder_group='unknown'")
    
    return disorder_map


def find_user_meta_path() -> str:
    """✅ 自动发现 user_meta 文件的实际路径"""
    for path in USER_META_PATH_CANDIDATES:
        if os.path.exists(path):
            return path
    
    # 搜索 intermediate_results 目录下的所有可能文件
    search_dir = os.path.dirname(PSM_MATCHED_PATH)
    if os.path.isdir(search_dir):
        for f in os.listdir(search_dir):
            if "user_meta" in f and f.endswith(".parquet"):
                return os.path.join(search_dir, f)
    
    return None


def main():
    # ======================== Step 1: 补全 PSM matched 文件 ========================
    print("=" * 70)
    print("Step 1: Fixing PSM matched file")
    print("=" * 70)
    
    if not os.path.exists(PSM_MATCHED_PATH):
        print(f"✗ PSM file not found: {PSM_MATCHED_PATH}")
    else:
        psm_df = pd.read_parquet(PSM_MATCHED_PATH)
        print(f"PSM shape: {psm_df.shape}")
        print(f"Columns: {list(psm_df.columns)}")
        
        # ✅ 修复: 正确识别患者
        if "is_patient" in psm_df.columns:
            is_patient_mask = psm_df["is_patient"] == 1
        elif "group" in psm_df.columns:
            is_patient_mask = psm_df["group"].isin(["patient"])
        else:
            is_patient_mask = pd.Series(False, index=psm_df.index)
            print("  ⚠️ Cannot identify patients!")
        
        patient_uids = set(psm_df.loc[is_patient_mask, "user_id_hash"])
        print(f"Total users: {len(psm_df)}, Patients: {len(patient_uids)}")
        
        need_fix = False
        
        if "disorder_group" in psm_df.columns:
            n_unknown = (psm_df.loc[is_patient_mask, "disorder_group"] == "unknown").sum()
            n_missing = psm_df.loc[is_patient_mask, "disorder_group"].isna().sum()
            if n_unknown == 0 and n_missing == 0:
                print("✓ All disorder_group values are known. No fix needed.")
            else:
                print(f"Found {n_unknown} unknown + {n_missing} NaN disorder_group values")
                need_fix = True
        else:
            print("disorder_group column missing, will add it")
            need_fix = True
        
        if need_fix:
            # ✅ 多源组合加载
            disorder_map = load_disorder_map_multi_source(patient_uids)
            
            # ✅ 修复: 用显式 mask 替代脆弱的 get() 链
            psm_df["disorder_group"] = np.where(
                is_patient_mask,
                psm_df["user_id_hash"].map(disorder_map).fillna("unknown"),
                "control"
            )
            
            # 统计
            print(f"\nDisorder group distribution (patients):")
            dg_counts = psm_df.loc[is_patient_mask, "disorder_group"].value_counts()
            for dg, count in dg_counts.items():
                pct = count / len(patient_uids) * 100
                print(f"  {dg:25s}: {count:7d} ({pct:.1f}%)")
            
            n_recovered = sum(psm_df.loc[is_patient_mask, "disorder_group"] != "unknown")
            print(f"\n  Recovery rate: {n_recovered}/{len(patient_uids)} ({n_recovered/len(patient_uids)*100:.1f}%)")
            
            # 备份 + 保存
            backup_path = PSM_MATCHED_PATH.replace(".parquet", "_before_disorder_fix.parquet")
            if not os.path.exists(backup_path):
                psm_df_original = pd.read_parquet(PSM_MATCHED_PATH)
                psm_df_original.to_parquet(backup_path, index=False)
                print(f"  Backup saved: {backup_path}")
            
            psm_df.to_parquet(PSM_MATCHED_PATH, index=False)
            print(f"  ✓ PSM file updated: {PSM_MATCHED_PATH}")
    
    # ======================== Step 2: 补全 user_meta 文件 ========================
    print("\n" + "=" * 70)
    print("Step 2: Fixing user_meta file")
    print("=" * 70)
    
    # ✅ 自动发现 user_meta 路径
    user_meta_path = find_user_meta_path()
    
    if user_meta_path is None:
        print("✗ user_meta file not found in any candidate location:")
        for p in USER_META_PATH_CANDIDATES:
            print(f"    {p}")
        print("\n  Searching directory for *user_meta* files...")
        search_dir = os.path.dirname(PSM_MATCHED_PATH)
        if os.path.isdir(search_dir):
            found = [f for f in os.listdir(search_dir) if "user_meta" in f]
            if found:
                print(f"  Found: {found}")
            else:
                print(f"  No *user_meta* files found in {search_dir}")
    else:
        print(f"Found user_meta: {user_meta_path}")
        
        um = pd.read_parquet(user_meta_path)
        print(f"  Shape: {um.shape}")
        print(f"  Columns: {list(um.columns)}")
        
        if "disorder_group" in um.columns:
            print("  disorder_group already exists in user_meta")
        else:
            # 优先从已修复的 PSM 文件获取
            disorder_map = {}
            
            if os.path.exists(PSM_MATCHED_PATH):
                try:
                    psm_check = pd.read_parquet(PSM_MATCHED_PATH, columns=["user_id_hash", "disorder_group"])
                    if "disorder_group" in psm_check.columns:
                        valid = psm_check[psm_check["disorder_group"] != "unknown"]
                        disorder_map = valid.set_index("user_id_hash")["disorder_group"].to_dict()
                        print(f"  Loaded {len(disorder_map)} mappings from PSM file")
                except Exception as e:
                    print(f"  Failed to read PSM file: {e}")
            
            # 如果 PSM 不够，从多源补充
            if "is_patient" in um.columns:
                patient_uids = set(um.loc[um["is_patient"] == 1, "user_id_hash"])
            elif "group" in um.columns:
                patient_uids = set(um.loc[um["group"].isin(["patient"]), "user_id_hash"])
            else:
                patient_uids = set()
            
            missing_uids = patient_uids - set(disorder_map.keys())
            if missing_uids:
                print(f"  Still missing {len(missing_uids)} patients, scanning additional sources...")
                extra_map = load_disorder_map_multi_source(missing_uids)
                disorder_map.update(extra_map)
            
            # 赋值
            if "is_patient" in um.columns:
                is_patient_mask = um["is_patient"] == 1
            elif "group" in um.columns:
                is_patient_mask = um["group"].isin(["patient"])
            else:
                is_patient_mask = pd.Series(False, index=um.index)
            
            um["disorder_group"] = np.where(
                is_patient_mask,
                um["user_id_hash"].map(disorder_map).fillna("unknown"),
                "control"
            )
            
            # 统计
            n_known = sum(um.loc[is_patient_mask, "disorder_group"] != "unknown")
            n_pat = is_patient_mask.sum()
            print(f"  Recovery: {n_known}/{n_pat} patients")
            
            # 备份 + 保存
            backup_path = user_meta_path.replace(".parquet", "_before_disorder_fix.parquet")
            if not os.path.exists(backup_path):
                pd.read_parquet(user_meta_path).to_parquet(backup_path, index=False)
                print(f"  Backup saved: {backup_path}")
            
            um.to_parquet(user_meta_path, index=False)
            print(f"  ✓ user_meta updated: {user_meta_path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()