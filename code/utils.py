# utils.py - Universal Utility Functions (Nature/Science Protocol)

import os
import io
import re
import json
import logging
import hashlib
import zstandard as zstd
import spacy
import demoji
import pandas as pd
from datetime import datetime
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Import final config
from config import LOG_LEVEL, LOG_FILE, PROCESSING_CONFIG, METADATA

class PresidioWarningFilter(logging.Filter):
    """Explicitly filter out the annoying ADDRESS warning from Presidio."""
    def filter(self, record):
        msg = record.getMessage()
        if "Entity ADDRESS doesn't have the corresponding recognizer" in msg:
            return False
        return True
# ======================== LOGGING CONFIGURATION ========================

def setup_presidio_logging():
    """屏蔽Presidio的所有WARNING/INFO级日志，只保留ERROR"""
    # 屏蔽Presidio根日志
    presidio_root_logger = logging.getLogger("presidio")
    presidio_root_logger.setLevel(logging.ERROR)
    presidio_root_logger.propagate = False  # 阻止日志向上传播
    
    # 兜底：屏蔽所有子模块
    for logger_name in [
        "presidio_analyzer",
        "presidio_analyzer.recognizer_registry",
        "presidio_analyzer.nlp_engine",
        "presidio_anonymizer",
        "presidio_evaluator"
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # 只保留CRITICAL级（几乎不会触发）
        logger.propagate = False  # 阻止日志向上传播到根日志器
       
# ======================== LOGGING CONFIGURATION ========================
def setup_logger():
    """Initialize reproducible logging (matches final config)"""
    os.makedirs(PROCESSING_CONFIG["output_dir"], exist_ok=True)
    
    # 1. Silence Third-Party Libraries
    logging.getLogger("presidio_analyzer").setLevel(logging.ERROR)
    logging.getLogger("presidio_anonymizer").setLevel(logging.ERROR)
    
    logger = logging.getLogger("reddit_cohort")
    logger.setLevel(LOG_LEVEL)
    
    if logger.handlers:
        return logger
    
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_path = os.path.join(PROCESSING_CONFIG["output_dir"], LOG_FILE)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    # Apply filter to file handler too
    file_handler.addFilter(PresidioWarningFilter())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.addFilter(PresidioWarningFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 3. Apply Filter to Root Logger (Catch-all for library warnings)
    root_logger = logging.getLogger()
    root_logger.addFilter(PresidioWarningFilter())
    
    return logger

logger = setup_logger()

# ======================== NLP TOOL INITIALIZATION ========================
def init_nlp_tools():
    """
    Lazy load NLP tools.
    CRITICAL: We keep the 'parser' enabled for Phase 1 dependency checks.
    """
    try:
        
        # SpaCy: Disable heavy components, but KEEP 'parser' for subject-verb analysis
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner", "vectors", "textcat"] 
        )
        
        # Presidio: Privacy/PII (Lazy load to save RAM if not needed)
        # Note: Presidio loads its own small spacy model internally
        from presidio_analyzer import RecognizerRegistry
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["en"])
        
        analyzer = AnalyzerEngine(
            registry=registry,
            supported_languages=["en"]  # 核心修复：用supported_languages而非language
        )
        anonymizer = AnonymizerEngine()
        # Emoji: Ensure codes are downloaded
        try:
            demoji.download_codes()
        except:
            pass # Handle offline cases if codes exist
            
        logger.info("NLP tools initialized. Parser: ENABLED. NER: DISABLED.")
        return nlp, analyzer, anonymizer
    except Exception as e:
        logger.error(f"NLP initialization failed. Run 'python -m spacy download en_core_web_sm' first. Error: {e}")
        raise

# def init_nlp_tools():
#     try:
#         # Disable heavy components for Spacy
#         nlp = spacy.load("en_core_web_sm", disable=["ner", "vectors", "textcat"])
        
#         # Presidio
#         # Set default score threshold to reduce noise
#         analyzer = AnalyzerEngine() 
#         anonymizer = AnonymizerEngine()
        
#         try:
#             demoji.download_codes()
#         except:
#             pass 
            
#         return nlp, analyzer, anonymizer
#     except Exception as e:
#         logger.error(f"NLP initialization failed. Error: {e}")
#         raise

# ======================== ZST FILE READER ========================

def read_zst_file(zst_path):
    """
    Generator that yields JSON objects from a ZST file.
    Added max_window_size=2147483648 to handle Pushshift's high compression.
    """
    try:
        with open(zst_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                for line in text_stream:
                    line = line.strip()
                    if not line: continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading {zst_path}: {e}")
        yield from []

# ======================== TEXT CLEANING STRATEGY ========================

def clean_text_fast(text):
    """
    [PHASE 2] FAST CLEANER.
    Removes URLs, Emails, and normalizes whitespace.
    Does NOT use Presidio (too slow for main loop).
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    # Remove Emails
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def clean_text_heavy(text, analyzer, anonymizer):
    """
    [PHASE 3 FINAL BUILD] COMPREHENSIVE REDACTION.
    Uses Presidio.
    Used ONLY for: Final Matched Dataset (Privacy)
    """
    if not isinstance(text, str) or not text:
        return ""
    try:
        # 1. Presidio (PII)
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
            language="en"
        )
        text = anonymizer.anonymize(text=text, analyzer_results=results).text
        
        # 2. Emoji
        text = demoji.replace_with_desc(text, sep=" ")
        
        # 3. Final Cleanup
        text = clean_text_fast(text)
        
        return text
    except Exception:
        return clean_text_fast(text)

# ======================== SECURITY & METADATA ========================

def hash_user_id(user_id):
    """SHA-256 + Salt hashing."""
    salt = PROCESSING_CONFIG["salt"]
    return hashlib.sha256((str(user_id) + salt).encode("utf-8")).hexdigest()

def save_metadata(additional_metadata=None):
    """Save metadata to JSON."""
    from config import BOT_THRESHOLDS, CONTROL_CONFIG, MENTAL_DISORDERS

    full_metadata = METADATA.copy()
    if additional_metadata:
        full_metadata.update(additional_metadata)

    full_metadata.update({
        "processing_config": PROCESSING_CONFIG,
        "bot_thresholds": BOT_THRESHOLDS,
        "control_config": CONTROL_CONFIG,
        "mental_disorders": MENTAL_DISORDERS
    })

    metadata_path = os.path.join(PROCESSING_CONFIG["output_dir"], PROCESSING_CONFIG["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=4)
    logger.info(f"Metadata saved: {metadata_path}")

def get_cpu_cores():
    cores = PROCESSING_CONFIG["cpu_cores"]
    if cores == -1:
        cores = max(1, os.cpu_count() - 1)
    return cores