import os
import glob
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from gensim.models import Word2Vec
from tqdm import tqdm

# CONFIG
SILVER_DIR = "/path/to/project/cohort_v1/silver_v2/data"
MODEL_PATH = "/path/to/project/cohort_v1/models/anchor_16y_stable_v2.model"
OUTPUT_DIR = "/path/to/project/cohort_v1/gold_v2/dimension_tables"

# MENTOR REQUIREMENT: Noise Filtering
MIN_POSTS = 100 
MIN_UNIQUE_USERS = 10 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_subreddit_embeddings():
    print("Loading Anchor Model...")
    # Load KeyVectors only (RAM efficient)
    w2v = Word2Vec.load(MODEL_PATH).wv
    
    print("Scanning Silver Layer for Subreddit Stats...")
    # We use PyArrow Dataset to aggregate without loading all text
    dataset = ds.dataset(SILVER_DIR, format="parquet")
    
    # We need to aggregate text by subreddit. 
    # Strategy: Iterate batches, accumulate word counts and vector sums.
    
    sub_stats = {} # {sub_name: {vec_sum: np.array, word_count: int, post_count: int, users: set}}
    
    # Streaming Read
    # Note: We need 'clean_text' to vectorize, 'user_id_hash' for confidence
    scanner = dataset.scanner(columns=["subreddit", "clean_text", "user_id_hash"], batch_size=50000)
    
    for batch in tqdm(scanner.to_batches(), desc="Vectorizing Subreddits"):
        df = batch.to_pandas()
        df = df.dropna(subset=["subreddit", "clean_text"])
        
        # Group by subreddit within this batch
        for sub, group in df.groupby("subreddit"):
            if sub not in sub_stats:
                sub_stats[sub] = {
                    "vector_sum": np.zeros(300), 
                    "valid_word_count": 0, 
                    "post_count": 0,
                    "users": set() # Use HLL sketch in production, set ok for <10M rows
                }
            
            # Vectorize text in this batch
            batch_vectors = []
            word_count_batch = 0
            
            # Simple vectorization: Average of words in batch
            # Optimized: Join all text, split, lookup
            all_text = " ".join(group["clean_text"].tolist()).lower().split()
            
            # Filter words present in model
            valid_vectors = [w2v[w] for w in all_text if w in w2v]
            
            if valid_vectors:
                vec_sum = np.sum(valid_vectors, axis=0)
                sub_stats[sub]["vector_sum"] += vec_sum
                sub_stats[sub]["valid_word_count"] += len(valid_vectors)
            
            sub_stats[sub]["post_count"] += len(group)
            sub_stats[sub]["users"].update(group["user_id_hash"].tolist())

    print("Finalizing Vectors & Filtering...")
    final_data = []
    
    for sub, stats in sub_stats.items():
        n_posts = stats["post_count"]
        n_users = len(stats["users"])
        n_words = stats["valid_word_count"]
        
        # MENTOR REQUIREMENT: Filter Low Activity
        if n_posts < MIN_POSTS or n_users < MIN_UNIQUE_USERS or n_words == 0:
            continue
            
        # Average
        avg_vec = stats["vector_sum"] / n_words
        
        final_data.append({
            "subreddit": sub,
            "embedding_vector": avg_vec.tolist(), # Store as list for Parquet
            "confidence_post_count": n_posts,
            "confidence_user_count": n_users,
            "confidence_word_count": n_words
        })
        
    df_out = pd.DataFrame(final_data)
    out_path = os.path.join(OUTPUT_DIR, "subreddit_embeddings_anchor_v2.parquet")
    df_out.to_parquet(out_path)
    print(f"Saved {len(df_out)} subreddit vectors to {out_path}")

if __name__ == "__main__":
    generate_subreddit_embeddings()