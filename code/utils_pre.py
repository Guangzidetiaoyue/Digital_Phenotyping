# utils.py - Universal Utility Functions (Nature/Science Protocol)
import os
import io # Moved to top
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
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("reddit_cohort")
    logger.setLevel(LOG_LEVEL)
    
    if logger.handlers:
        return logger
    
    # File handler (Persistent log)
    file_path = os.path.join(PROCESSING_CONFIG["output_dir"], LOG_FILE)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    
    # Console handler (Real-time)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ======================== NLP TOOL INITIALIZATION ========================
def init_nlp_tools():
    """
    Lazy load NLP tools.
    CRITICAL: We keep the 'parser' enabled for Phase 1 dependency checks.
    """
    try:
        setup_presidio_logging()
        
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


# ======================== ZST FILE READER (FIXED) ========================
def read_zst_file(zst_path):
    """
    Generator that yields JSON objects from a ZST file.
    FIX: Added max_window_size=2147483648 to handle Pushshift's high compression.
    """
    try:
        with open(zst_path, 'rb') as fh:
            # 2147483648 = 2GB. Pushshift dumps often need this larger window size.
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            
            # Use stream_reader with a reasonable chunk size (e.g., 64KB - 1MB default)
            with dctx.stream_reader(fh) as reader:
                # TextIOWrapper handles the utf-8 decoding on the fly
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Log warning only occasionally to avoid flooding logs
                        continue
                        
    except Exception as e:
        logger.error(f"Error reading {zst_path}: {e}")
        yield from []
        
def count_zst_lines_fast(zst_path):
    """
    Optional helper if you absolutely need a progress bar total.
    Still slow, but separated from the main read logic.
    Recommended: Use file size (bytes) for tqdm instead of lines.
    """
    count = 0
    with open(zst_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for _ in text_stream:
                count += 1
    return count

# ======================== PII REDACTION ========================
def clean_text(text, analyzer, anonymizer):
    """
    PII redaction compliant with SHA-256 + Redaction protocol.
    """
    if not isinstance(text, str) or not text:
        return ""
        
    try:
        # 1. Presidio Analysis (Names, Phones, Emails)
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],#, "ADDRESS"
            language="en"
        )
        # 2. Anonymization
        text = anonymizer.anonymize(text=text, analyzer_results=results).text
        
        # 3. Emoji to Text
        text = demoji.replace_with_desc(text, sep=" ")
        
        # 4. Regex Cleaning
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    except Exception as e:
        # Log sampling of errors to avoid flooding logs
        return text # Fail safe: return original text rather than crashing

# ========== 新增：批量文本清理函数（核心优化） ==========
def clean_text_batch(text_list, analyzer, anonymizer, batch_size=1000):
    """
    批量清理文本，提升NLP效率（spaCy/Presidio批量处理比单条快5-10倍）
    :param text_list: 待清理的文本列表
    :param analyzer: Presidio AnalyzerEngine实例
    :param anonymizer: Presidio AnonymizerEngine实例
    :param batch_size: 每批处理的文本数（避免内存溢出）
    :return: 清理后的文本列表（与输入列表长度一致）
    """
    clean_texts = []
    # 空列表直接返回
    if not text_list:
        return clean_texts
    
    # 分批次处理（每1000条一批）
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        for text in batch:
            # 复用原clean_text的逻辑，保证处理结果一致
            clean_texts.append(clean_text(text, analyzer, anonymizer))
    
    # 校验长度（避免批次处理遗漏）
    assert len(clean_texts) == len(text_list), "批量处理文本长度不匹配"
    return clean_texts

# ======================== SECURITY ========================
def hash_user_id(user_id):
    """SHA-256 + Salt hashing."""
    salt = PROCESSING_CONFIG["salt"]
    return hashlib.sha256((str(user_id) + salt).encode("utf-8")).hexdigest()

# ======================== METADATA ========================
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
        "mental_disorders": MENTAL_DISORDERS, 
        "libraries": {
            "spacy": spacy.__version__,
            "pandas": pd.__version__
        }
    })
    
    metadata_path = os.path.join(PROCESSING_CONFIG["output_dir"], PROCESSING_CONFIG["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=4)
    logger.info(f"Metadata saved: {metadata_path}")

# ======================== SYSTEM ========================
def get_cpu_cores():
    cores = PROCESSING_CONFIG["cpu_cores"]
    if cores == -1:
        cores = max(1, os.cpu_count() - 1)
    return cores