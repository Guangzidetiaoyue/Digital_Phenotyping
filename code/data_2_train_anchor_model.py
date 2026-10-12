import os
import glob
import json
import logging
import psutil
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pyarrow.parquet as pq

# CONFIG
# Using the Corpus Layer (Pre-tokenized) for speed
CORPUS_DIR = "/path/to/project/cohort_v1/corpus_layer"
MODEL_DIR = "/path/to/project/cohort_v1/models"
ANCHOR_NAME = "anchor_16y_stable_v2"

# MENTOR REQUIREMENT: High min_count to filter noise/typos for the "Coordinate System"
ANCHOR_MIN_COUNT = 100  
VECTOR_SIZE = 300
WINDOW = 5

os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class CorpusIter:
    """Streams tokenized data from Parquet."""
    def __init__(self, year_filter=None):
        self.files = glob.glob(os.path.join(CORPUS_DIR, "*.parquet"))
        self.year_filter = year_filter
        
    def __iter__(self):
        for f in self.files:
            # Efficient read
            try:
                t = pq.read_table(f, columns=["tokens_basic", "year"])
                df = t.to_pandas()
                if self.year_filter:
                    df = df[df["year"] == self.year_filter]
                
                # Yield non-empty token lists
                for tokens in df["tokens_basic"]:
                    if len(tokens) > 0:
                        yield tokens.tolist()
            except Exception as e:
                logging.error(f"Read error {f}: {e}")

class TrainingMonitor(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        mem = psutil.virtual_memory().percent
        logging.info(f"Epoch {self.epoch} finished. RAM: {mem}%. Vocab Size: {len(model.wv)}")
        self.epoch += 1

def train_anchor():
    logging.info(f"Initializing Anchor Model (Min Count: {ANCHOR_MIN_COUNT})...")
    
    # 1. Train Anchor (All Years)
    sentences = CorpusIter()
    model = Word2Vec(
        vector_size=VECTOR_SIZE, 
        window=WINDOW, 
        min_count=ANCHOR_MIN_COUNT, 
        workers=16, 
        sg=1, # Skip-gram is scientifically superior for semantics
        seed=42
    )
    
    logging.info("Building Anchor Vocabulary...")
    model.build_vocab(sentences)
    
    logging.info("Training Anchor...")
    model.train(sentences, total_examples=model.corpus_count, epochs=3, callbacks=[TrainingMonitor()])
    
    # Save Model
    path = os.path.join(MODEL_DIR, f"{ANCHOR_NAME}.model")
    model.save(path)
    
    # MENTOR REQUIREMENT: Log Vocabulary Stats for Intersection Checks
    vocab_keys = list(model.wv.index_to_key)
    stats = {
        "total_vocab": len(vocab_keys),
        "min_count": ANCHOR_MIN_COUNT,
        "sample_words": vocab_keys[:50] # Top 50 frequent words
    }
    with open(os.path.join(MODEL_DIR, f"{ANCHOR_NAME}_vocab_log.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    logging.info("Anchor Model Trained & Logs Saved.")

if __name__ == "__main__":
    train_anchor()