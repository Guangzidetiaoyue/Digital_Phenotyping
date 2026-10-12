import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# CONFIG
# Input: The Robust Silver Layer we built in Stage 1
SILVER_DIR = "/path/to/project/cohort_v1/silver_v2/data"
# Output: The Tokenized Corpus Layer (Input for Word2Vec)
CORPUS_DIR = "/path/to/project/cohort_v1/corpus_layer"

os.makedirs(CORPUS_DIR, exist_ok=True)

def process_silver_partition(file_path):
    """
    Reads a Silver Partition (Raw Text) -> Writes a Corpus Partition (Tokens).
    """
    file_name = os.path.basename(file_path)
    # Naming convention: corpus_part_0000.parquet
    out_name = file_name.replace("part_", "corpus_")
    out_path = os.path.join(CORPUS_DIR, out_name)
    
    # Skip if already exists (Breakpoint continuation)
    if os.path.exists(out_path):
        return

    try:
        # 1. Read only necessary columns to save RAM
        df = pd.read_parquet(
            file_path, 
            columns=["user_id_hash", "clean_text", "post_timestamp", "group"]
        )
        
        # 2. Extract Temporal Metadata (For time-sliced training later)
        df["dt"] = pd.to_datetime(df["post_timestamp"], unit='s')
        df["year"] = df["dt"].dt.year.astype("int16") # int16 saves space
        
        # 3. Tokenization (The expensive CPU step)
        # We save "tokens_basic": Lowercase, split by whitespace.
        # This preserves everything. W2V min_count will handle noise later.
        df["tokens_basic"] = df["clean_text"].fillna("").astype(str).str.lower().str.split()
        
        # 4. Filter Empty (Reduces file size significantly)
        # We only keep rows that have actual words for embedding training
        df_corpus = df[df["tokens_basic"].apply(len) > 0].copy()
        
        if df_corpus.empty:
            return

        # 5. Save as Optimized Parquet
        # Schema: user_id (for linkage), year (for slicing), group (for contrast), tokens (for training)
        table = pa.Table.from_pandas(
            df_corpus[["user_id_hash", "year", "group", "tokens_basic"]],
            preserve_index=False
        )
        
        pq.write_table(table, out_path, compression='snappy')
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Get all Silver partitions
    files = sorted(glob.glob(os.path.join(SILVER_DIR, "*.parquet")))
    
    print(f"Preparing Corpus from {len(files)} partitions...")
    print(f"Input: {SILVER_DIR}")
    print(f"Output: {CORPUS_DIR}")
    
    # Use sequential processing or simple loop to avoid Memory spikes
    # (Tokenization creates many small strings, which can bloat RAM in multiprocessing)
    for f in tqdm(files, desc="Tokenizing"):
        process_silver_partition(f)
        
    print("Corpus Layer Prepared. You can now run train_anchor_model.py")