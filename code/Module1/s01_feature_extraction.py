"""
Step 01: Four-Dimensional Feature Extraction (Production-Grade v1.2)

Changelog from v1.1:
  [FIX] Present focus score: added RE_PRESENT_TENSE (was n_sentences - past - future)
  [FIX] Restructured extraction: always compute regex base, then override with LIWC/NRC
  [FIX] NRC path: now inherits ALL regex features for non-emotion dimensions
  [FIX] NRC reverse index: O(W) lookup instead of O(W*E)
  [FIX] Social jetlag: circular distance for hour comparison
  [FIX] NETSPEAK regex: separated emoji/emoticon from word patterns
  [FIX] File naming: consistent output for Step 02 compatibility
  [FIX] Removed Z-score generation (belongs in Step 02)
  [NEW] VADER truncation ratio tracking
  [NEW] User-level features: emotion inertia, post length CV, pronoun variability
  [NEW] MIN_POSTS_FOR_TEMPORAL parameter
  [OPT] Memory management: reduced per-partition memory footprint
  [OPT] User aggregation: vectorized groupby instead of manual loops
"""
# ============================================================
# [路径自修复] 确保本脚本被其他模块导入时，仍能找到同级目录的配置文件
# 当本脚本被直接运行时，Python 自动添加本目录到 sys.path；
# 但被 from Module1.s01... 导入时不会，因此需要手动添加。
# ============================================================
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import gc
import re
import json
import logging
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from math import pi
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from scipy.stats import entropy, kurtosis, skew
from datetime import datetime

from config_module1 import (
    SILVER_DATA_DIR, FEATURES_DIR, PSM_MATCHED_PATH,
    MENTAL_HEALTH_SUBREDDITS, SUBREDDIT_CATEGORIES,
    NUM_CORES, BATCH_SIZE, MIN_POSTS_FOR_FEATURES, MIN_WORDS_FOR_LINGUISTIC,
    MIN_POSTS_FOR_TEMPORAL, MATTR_MAX_WORDS, RANDOM_SEED,
    NRC_LEXICON_PATH, USE_LIWC, LIWC_DICTIONARY_PATH,
    SOURCE_POLICY, RAW_FEATURES_FILE, FEATURE_METADATA_FILE, QUALITY_REPORT_FILE
)



# ======================== SETUP & LOGGING ========================

os.makedirs(FEATURES_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(FEATURES_DIR, "extraction.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(RANDOM_SEED)


# ======================== REGEX PATTERNS (Complete Set) ========================

# Level 1: Text Cleaning
RE_WORDS = re.compile(r'\b\w+\b')
RE_URL = re.compile(r'<URL>|https?://\S+|www\.\S+')
RE_EMAIL = re.compile(r'<EMAIL>|\S+@\S+')
RE_SENTENCES = re.compile(r'[.!?]+')

# Level 2: Punctuation & Behavioral Markers
RE_QUESTION = re.compile(r'\?')
RE_EXCLAMATION = re.compile(r'!')
RE_QUOTATION = re.compile(r'["\']')
RE_ELLIPSIS = re.compile(r'\.{2,}')

# Level 3: Pronouns
RE_FIRST_PERSON = re.compile(
    r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b',
    re.IGNORECASE
)
RE_SECOND_PERSON = re.compile(
    r'\b(you|your|yours|yourself|yourselves)\b',
    re.IGNORECASE
)
RE_THIRD_PERSON = re.compile(
    r'\b(he|she|it|they|him|her|its|them|his|hers|their|theirs|'
    r'themselves|himself|herself|itself)\b',
    re.IGNORECASE
)
RE_IMPERSONAL_PRONOUN = re.compile(
    r'\b(this|that|these|those|'
    r'anything|something|nothing|everything)\b',
    re.IGNORECASE
)

# Temporal Markers — Past Tense
RE_PAST_TENSE = re.compile(r'''
    \b(
    # 不规则动词保持原样
    said|made|saw|came|took|got|found|thought|knew|felt|
    told|became|left|put|meant|kept|let|began|stayed|heard|spoke|
    read|wrote|stood|lost|paid|met|sat|ran|brought|caught|bought|
    taught|slept|drove|ate|drank|went|had|was|were|did|had\s+been|
    has\s+been|used\s+to|didn\'t|wasn\'t|weren\'t|hadn\'t|
    # 补充高频规则动词过去式（替代 \w+ed）
    looked|wanted|needed|tried|started|liked|loved|helped|turned|
    worked|played|changed|moved|lived|seemed|felt|happened|stopped|
    talked|asked|watched|followed|created|considered|decided|
    believed|remembered|supposed|noticed|realized|checked
    )\b
''', re.IGNORECASE | re.VERBOSE)

# [FIX] Temporal Markers — Present Tense (NEW)
# Replaces the broken calculation: n_sentences - past - future
RE_PRESENT_TENSE = re.compile(
    r'\b('
    # Be verbs (present)
    r'am|is|are|'
    # Do/Have auxiliaries (present)
    r'do|does|have|has|'
    # Common present-tense base form verbs
    r'feel|think|know|want|need|make|take|get|go|come|see|say|'
    r'find|keep|let|seem|look|give|tell|use|try|leave|call|'
    r'work|play|run|walk|live|love|hate|like|believe|hope|wish|'
    r'mean|understand|remember|forget|worry|cry|laugh|smile|hurt|'
    r'stay|wait|sit|stand|listen|watch|read|write|eat|drink|'
    r'sleep|wake|start|stop|help|show|hear|learn|turn|put|'
    r'bring|hold|fall|talk|ask|answer|explain|follow|move|change|'
    # Common 3rd person present (-s/-es/-ies)
    r'feels|thinks|knows|wants|needs|makes|takes|gets|goes|comes|'
    r'sees|says|finds|keeps|seems|looks|gives|tells|uses|tries|'
    r'leaves|calls|works|plays|runs|walks|lives|loves|hates|likes|'
    r'believes|hopes|wishes|means|understands|remembers|forgets|'
    r'worries|cries|laughs|smiles|hurts|stays|waits|sits|stands|'
    r'listens|watches|reads|writes|eats|drinks|sleeps|wakes|'
    r'starts|stops|helps|shows|hears|learns|turns|puts|'
    r'brings|holds|falls|talks|asks|answers|explains|follows|moves|changes'
    r')\b',
    re.IGNORECASE
)

# Temporal Markers — Future Tense
RE_FUTURE_TENSE = re.compile(
    r'\b(will|shall|gonna|going\s+to|\'ll|'
    r'will\s+be|going\s+to\s+(?:be|have|do|take|make|get))\b',
    re.IGNORECASE
)

# Negation
RE_NEGATION = re.compile(
    r"\b(no|not|never|none|no\s+one|nobody|nothing|nowhere|neither|nor|"
    r"can't|cannot|won't|wouldn't|shouldn't|couldn't|mustn't|didn't|"
    r"doesn't|don't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"aren't|isn't|mightn't|needn't|shan't|ain't)\b",
    re.IGNORECASE
)

# Swearing
RE_SWEAR = re.compile(
    r'\b(fuck|fucking|fucked|fuckin|shit|damn|dammit|hell|bitch|bastard|'
    r'ass|asshole|crap|damned|goddamn|wtf|stfu|wth)\b',
    re.IGNORECASE
)

# Clinical & Psychological Constructs
RE_DEATH_WORDS = re.compile(
    r'\b(death|die|dying|dead|deceased|killed|kill|killing|suicide|'
    r'suicidal|murder|murdered|overdose|OD|end\s+(?:it|all|my\s+life)|'
    r'kms|kys|unalive|suicid|suicidal\s+ideation)\b',
    re.IGNORECASE
)

RE_ANGER_WORDS = re.compile(
    r'\b(angry|anger|hate|hatred|furious|rage|raging|hostile|annoyed|'
    r'irritated|frustrated|resentment|bitter|pissed|pissed\s+off|'
    r'mad\s+(?:at|about)|irate|livid|infuriated|aggravated)\b',
    re.IGNORECASE
)

RE_ANXIETY_WORDS = re.compile(
    r'\b(anxious|anxiety|worried|worry|worrying|nervous|panic|panicking|'
    r'scared|afraid|fear|terrified|dread|uneasy|apprehensive|tense|'
    r'restless|phobia|overthinking|racing\s+thoughts|nervousness|'
    r'worrywart|anxiety\s+attack|panic\s+attack)\b',
    re.IGNORECASE
)

RE_SADNESS_WORDS = re.compile(
    r'\b(sad|sadness|depressed|depressing|depression|hopeless|helpless|'
    r'miserable|lonely|loneliness|grief|mourning|heartbroken|cry|'
    r'crying|tears|sobbing|emptiness|empty|numb|void|despair|'
    r'devastated|disappointed|blue|down|hard\s+time)\b',
    re.IGNORECASE
)

RE_POSITIVE_WORDS = re.compile(
    r'\b(happy|happiness|joy|joyful|glad|cheerful|excited|wonderful|'
    r'amazing|fantastic|great|love|loving|grateful|thankful|blessed|'
    r'beautiful|hope|hopeful|optimistic|proud|confident|relief|'
    r'celebrating|achieved|awesome|excellent|best|win|winning|won|'
    r'successful|success)\b',
    re.IGNORECASE
)

# Cognitive Processes
RE_CAUSAL = re.compile(
    r'\b(because|cause|since|therefore|hence|thus|consequently|'
    r'as\s+a\s+result|due\s+to|owing\s+to|result\s+in|results\s+in|'
    r'lead\s+to|leads\s+to|led\s+to|caused\s+by|triggered|'
    r'provoke|provoked|effect|affect)\b',
    re.IGNORECASE
)

RE_INSIGHT = re.compile(
    r'\b(thinks?|knows?|consider|understand|realize|recognize|'
    r'believe|feels?|sense|mean|notice|thought|aware|insight|'
    r'epiphany|comprehend|grasp|conclude|realization)\b',
    re.IGNORECASE
)

RE_DISCREPANCY = re.compile(
    r'\b(ought|unexpected|surprising|strange|weird|odd|unusual|'
    r'different|difference|changed|change|shifting|shift)\b',
    re.IGNORECASE
)


RE_TENTATIVE = re.compile(
    r'\b(maybe|perhaps|possibly|might|could|would|should|'
    r'seem|seems|appear|guess|suppose|probably|likely|unlikely|'
    r'almost|sort\s+of|kind\s+of|somewhat|fairly|rather|'
    r'presumably|tentative|uncertain|unsure)\b',
    re.IGNORECASE
)

RE_CERTAINTY = re.compile(
    r'\b(always|never|absolutely|definitely|certainly|surely|'
    r'clearly|undoubtedly|completely|totally|entirely|must|'
    r'always|never|exactly|precisely|certain|sure|positive|'
    r'without\s+doubt|indeed|obviously|evidently)\b',
    re.IGNORECASE
)

RE_INHIBITION = re.compile(
    r'\b(avoid|stop|block|restrain|refrain|eats?|swallow|'
    r'consumed|consume|give\s+up|giving\s+up|prevent|cease|'
    r'inhibit|restraint|refrained|stopped|quit)\b',
    re.IGNORECASE
)

# [FIX] NETSPEAK: separated emoji/emoticon from word-boundary patterns
RE_NETSPEAK_WORDS = re.compile(
    r'\b(lol|lmao|rofl|omg|wtf|btw|imo|imho|tbh|fyi|'
    r'irl|tbqh|rn|bc|cuz|uh|uhh|um|umm|haha|hehe|'
    r'xD|XD)\b',
    re.IGNORECASE
)

RE_NETSPEAK_EMOJI = re.compile(
    r'😂|👍|❤️|💔|:\)|:\(|;\)|<3',
)

# Assent & Nonfluencies & Fillers
RE_ASSENT = re.compile(
    r'\b(yes|yeah|yep|yup|okay|ok|sure|absolutely|'
    r'indeed|of\s+course|agree|agreed|right|correct|'
    r'totally|definitely|sure\s+thing)\b',
    re.IGNORECASE
)

RE_NONFLUENCIES = re.compile(
    r'\b(uh|um|er|ah|eh|uhm|umm|hmm|huh)\b',
    re.IGNORECASE
)

RE_FILLER = re.compile(
    r'\b(you\s+know|i\s+mean|like|sorta|kinda|'
    r'basically|literally|actually|honestly|right\?)\b',
    re.IGNORECASE
)


# ======================== UTILITY FUNCTIONS ========================

def calc_mattr(words: List[str], window_size: int = 50) -> float:
    """
    Moving-Average Type-Token Ratio (Covington & McFall, 2010)
    [FIX] Capped computation for very long texts
    """
    if not words or len(words) < 10:
        return np.nan

    # [FIX] Sample for very long texts to avoid O(n²) blowup
    if len(words) > MATTR_MAX_WORDS:
        rng = np.random.RandomState(RANDOM_SEED)
        indices = sorted(rng.choice(len(words), size=MATTR_MAX_WORDS, replace=False))
        words = [words[i] for i in indices]

    if len(words) <= window_size:
        return len(set(words)) / len(words)

    ttrs = []
    # step = max(1, window_size // 2)
    for i in range(0, len(words) - window_size + 1): #, step
        window = words[i:i + window_size]
        ttrs.append(len(set(window)) / len(window))

    return np.mean(ttrs) if ttrs else np.nan


def safe_divide(numerator: float, denominator: float, default: float = np.nan) -> float:
    """Safe division with zero/NaN protection"""
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return numerator / denominator


def signed_circular_diff(h1: float, h2: float) -> float:
    """
    带符号的圆形距离：h1 相对于 h2 的偏移。
    正值表示 h1 在 h2 之后（相位延迟），负值表示 h1 在 h2 之前（相位提前）。
    例：signed_circular_diff(2, 23) = 3 (推迟3小时)
        signed_circular_diff(22, 1) = -3 (提前3小时)
    """
    diff = h1 - h2
    if diff > 12:
        diff -= 24
    elif diff < -12:
        diff += 24
    return diff


# ======================== DICTIONARY CLASSES ========================

class LIWCDictionary:
    """
    LIWC 2015/2022 Dictionary Loader
    Parses .dic format with correct section handling.
    """
    def __init__(self, dict_path: str):
        self.categories: Dict[str, str] = {}       # cat_id -> cat_name
        self.words: Dict[str, Set[str]] = {}        # exact word -> cat_ids
        self.wildcards: Dict[str, Set[str]] = {}    # prefix -> cat_ids

        self._load(dict_path)

        # Reverse lookup: cat_name -> words (for diagnostics)
        self.cat_to_words: Dict[str, Set[str]] = defaultdict(set)
        for word, cats in self.words.items():
            for cat in cats:
                if cat in self.categories:
                    self.cat_to_words[self.categories[cat]].add(word)
                else:
                    self.cat_to_words[cat].add(word)

        logger.info(f"LIWC Dictionary loaded: {len(self.categories)} categories, "
                    f"{len(self.words)} exact words, {len(self.wildcards)} wildcards")

        # Validate essential categories
        essential = {'anx', 'anger', 'sad', 'posemo', 'negemo', 'i', 'we', 'you',
                     'shehe', 'they', 'insight', 'cause', 'tentat', 'certain',
                     'focuspast', 'focuspresent', 'focusfuture', 'death', 'swear',
                     'netspeak', 'social', 'family', 'friend', 'inhib'}
        available = set(self.categories.values())
        missing = essential - available
        if missing:
            logger.warning(f"LIWC missing essential categories: {missing}")

    def _load(self, path: str):
        section = 0
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if line == '%':
                        section += 1
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if section == 1:
                        if parts[0].isdigit():
                            cat_id = parts[0]
                            cat_name = parts[1] if len(parts) > 1 else f"cat_{cat_id}"
                            self.categories[cat_id] = cat_name

                    elif section == 2:
                        if len(parts) < 2:
                            continue
                        word = parts[0].lower()
                        cat_ids = {c for c in parts[1:] if c.isdigit()}
                        if not cat_ids:
                            continue
                        if word.endswith('*'):
                            self.wildcards[word[:-1]] = cat_ids
                        else:
                            self.words[word] = cat_ids
        except Exception as e:
            logger.error(f"Error loading LIWC dictionary from {path}: {e}")
            raise

    def lookup(self, token: str) -> Set[str]:
        """Multi-tier lookup: exact -> wildcard prefix (longest match)"""
        token = token.lower()
        if token in self.words:
            return self.words[token]
        for i in range(len(token), 0, -1):
            prefix = token[:i]
            if prefix in self.wildcards:
                return self.wildcards[prefix]
        return set()

    def get_category_counts(self, tokens: List[str]) -> Dict[str, int]:
        """Count occurrences per category (by category name)"""
        counts = defaultdict(int)
        for token in tokens:
            if not token or len(token) < 1:
                continue
            cat_ids = self.lookup(token)
            for cat_id in cat_ids:
                if cat_id in self.categories:
                    counts[self.categories[cat_id]] += 1
                else:
                    counts[f"unknown_{cat_id}"] += 1
        return dict(counts)

    def get_coverage_report(self) -> Dict[str, any]:
        return {
            'n_categories': len(self.categories),
            'n_exact_words': len(self.words),
            'n_wildcards': len(self.wildcards),
            'category_sample': dict(list(self.categories.items())[:10]),
            'word_sample': dict(list(self.words.items())[:5]),
            'wildcard_sample': dict(list(self.wildcards.items())[:5])
        }


class NRCLexicon:
    """
    [FIX] NRC Emotion Lexicon with reverse index for O(W) lookup.
    Original: O(W × E) with nested loops.
    New: O(W) with word -> emotions reverse mapping.
    """
    def __init__(self, lexicon_path: str):
        # Forward: emotion -> word set
        self.emotion_to_words: Dict[str, Set[str]] = defaultdict(set)
        # [FIX] Reverse: word -> emotion set (for fast lookup)
        self.word_to_emotions: Dict[str, Set[str]] = defaultdict(set)
        self.emotions: List[str] = []

        self._load(lexicon_path)

        logger.info(f"NRC Lexicon loaded: {len(self.emotions)} emotions, "
                    f"{len(self.word_to_emotions)} unique words")

    def _load(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3 and parts[2] == '1':
                        word = parts[0].lower()
                        emotion = parts[1]
                        self.emotion_to_words[emotion].add(word)
                        self.word_to_emotions[word].add(emotion)

            self.emotions = sorted(self.emotion_to_words.keys())
        except Exception as e:
            logger.error(f"Error loading NRC lexicon from {path}: {e}")
            raise

    def get_emotion_counts(self, tokens: List[str]) -> Dict[str, int]:
        """
        [FIX] O(W) lookup using reverse index.
        Each token is looked up once in the reverse dictionary.
        """
        counts = defaultdict(int)
        for token in tokens:
            if token in self.word_to_emotions:
                for emotion in self.word_to_emotions[token]:
                    counts[emotion] += 1
        return dict(counts)


class DictionaryManager:
    """
    Singleton-style manager for linguistic dictionaries.
    Handles initialization and provides process-safe access.
    """
    _instance: Optional['DictionaryManager'] = None
    _liwc: Optional[LIWCDictionary] = None
    _nrc: Optional[NRCLexicon] = None
    _vader = None
    _initialized: bool = False
    # [FIX] Track the effective source for the entire run
    _effective_source: str = "unknown"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, force_reload: bool = False):
        """Initialize all dictionaries - call once per process"""
        if self._initialized and not force_reload:
            return

        logger.info(f"Initializing dictionaries in PID {os.getpid()}")

        # VADER
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                logger.info("VADER initialized")
            except ImportError:
                logger.warning("VADER not available")

        # LIWC
        liwc_loaded = False
        if USE_LIWC and os.path.exists(LIWC_DICTIONARY_PATH):
            try:
                self._liwc = LIWCDictionary(LIWC_DICTIONARY_PATH)
                liwc_loaded = True
                self._effective_source = "liwc"
                logger.info("LIWC initialized")
            except Exception as e:
                logger.error(f"LIWC initialization failed: {e}")
                self._liwc = None

        # NRC
        nrc_loaded = False
        if self._nrc is None and os.path.exists(NRC_LEXICON_PATH):
            try:
                self._nrc = NRCLexicon(NRC_LEXICON_PATH)
                nrc_loaded = True
                if not liwc_loaded:
                    self._effective_source = "nrc"
                logger.info("NRC initialized")
            except Exception as e:
                logger.error(f"NRC initialization failed: {e}")
                self._nrc = None

        if not liwc_loaded and not nrc_loaded:
            self._effective_source = "regex"

        self._initialized = True

        # [FIX] Strict source policy check
        if SOURCE_POLICY == "strict":
            if USE_LIWC and not liwc_loaded:
                logger.error(
                    "STRICT MODE: LIWC enabled but failed to load. "
                    "Set USE_LIWC=False or SOURCE_POLICY='permissive' to continue."
                )
                # In strict mode, we still continue but log the inconsistency
                logger.warning("Continuing with fallback source — results may be inconsistent")

        logger.info(f"Effective source for this process: {self._effective_source}")

    @property
    def liwc(self) -> Optional[LIWCDictionary]:
        return self._liwc

    @property
    def nrc(self) -> Optional[NRCLexicon]:
        return self._nrc

    @property
    def vader(self):
        return self._vader

    @property
    def effective_source(self) -> str:
        return self._effective_source

    def get_status(self) -> Dict[str, any]:
        return {
            'liwc_available': self._liwc is not None,
            'nrc_available': self._nrc is not None,
            'vader_available': self._vader is not None,
            'effective_source': self._effective_source,
            'pid': os.getpid()
        }


# Module-level singleton
_dict_manager = DictionaryManager()


def init_worker():
    """Worker initialization - MUST be called in each process"""
    _dict_manager.initialize()


def get_liwc() -> Optional[LIWCDictionary]:
    if not _dict_manager._initialized:
        _dict_manager.initialize()
    return _dict_manager.liwc


def get_nrc() -> Optional[NRCLexicon]:
    if not _dict_manager._initialized:
        _dict_manager.initialize()
    return _dict_manager.nrc


def get_vader():
    if not _dict_manager._initialized:
        _dict_manager.initialize()
    return _dict_manager.vader


def get_effective_source() -> str:
    if not _dict_manager._initialized:
        _dict_manager.initialize()
    return _dict_manager.effective_source


# ======================== FEATURE EXTRACTION ENGINE ========================

def _compute_regex_features(clean: str, words_lower: List[str],
                            n_words: int, n_sentences: int) -> Dict:
    """
    [FIX] Compute ALL regex-based features as a complete base layer.
    This is ALWAYS called, regardless of LIWC/NRC availability.
    Ensures no features are missing when NRC is the emotion source.
    """
    return {
        # Affective
        'anxiety_score': len(RE_ANXIETY_WORDS.findall(clean)),
        'anger_score': len(RE_ANGER_WORDS.findall(clean)),
        'sadness_score': len(RE_SADNESS_WORDS.findall(clean)),
        'positive_score': len(RE_POSITIVE_WORDS.findall(clean)),
        'negative_score': (len(RE_ANXIETY_WORDS.findall(clean)) +
                          len(RE_ANGER_WORDS.findall(clean)) +
                          len(RE_SADNESS_WORDS.findall(clean))),

        # Pronouns
        'first_person_score': len(RE_FIRST_PERSON.findall(clean)),
        'second_person_score': len(RE_SECOND_PERSON.findall(clean)),
        'third_person_score': len(RE_THIRD_PERSON.findall(clean)),
        'impersonal_score': len(RE_IMPERSONAL_PRONOUN.findall(clean)),

        # Cognitive
        'insight_score': len(RE_INSIGHT.findall(clean)),
        'causal_score': len(RE_CAUSAL.findall(clean)),
        'certainty_score': len(RE_CERTAINTY.findall(clean)),
        'tentative_score': len(RE_TENTATIVE.findall(clean)),
        'discrepancy_score': len(RE_DISCREPANCY.findall(clean)),
        'inhibition_score': len(RE_INHIBITION.findall(clean)),

        # Temporal focus
        # [FIX] Use RE_PRESENT_TENSE instead of n_sentences - past - future
        'past_focus_score': len(RE_PAST_TENSE.findall(clean)),
        'present_focus_score': len(RE_PRESENT_TENSE.findall(clean)),
        'future_focus_score': len(RE_FUTURE_TENSE.findall(clean)),

        # Risk & Other
        'death_score': len(RE_DEATH_WORDS.findall(clean)),
        'swear_score': len(RE_SWEAR.findall(clean)),
        'assent_score': len(RE_ASSENT.findall(clean)),
        'nonfluency_score': len(RE_NONFLUENCIES.findall(clean)),
    }


def _compute_liwc_overrides(words_lower: List[str]) -> Tuple[Dict, Dict[str, str]]:
    """
    Compute LIWC features that OVERRIDE the regex base layer.
    Returns: (feature_overrides, source_map)
    """
    liwc = get_liwc()
    if liwc is None:
        return {}, {}

    counts = liwc.get_category_counts(words_lower)

    overrides = {
        # Affective
        'anxiety_score': counts.get('anx', 0),
        'anger_score': counts.get('anger', 0),
        'sadness_score': counts.get('sad', 0),
        'positive_score': counts.get('posemo', 0),
        'negative_score': counts.get('negemo', 0),

        # Pronouns
        'first_person_score': counts.get('i', 0) + counts.get('we', 0),
        'second_person_score': counts.get('you', 0),
        'third_person_score': counts.get('shehe', 0) + counts.get('they', 0),
        'impersonal_score': counts.get('ipron', 0),

        # Cognitive
        'insight_score': counts.get('insight', 0),
        'causal_score': counts.get('cause', 0),
        'certainty_score': counts.get('certain', 0),
        'tentative_score': counts.get('tentat', 0),
        'discrepancy_score': counts.get('discrep', 0),
        'inhibition_score': counts.get('inhib', 0) or counts.get('inhibit', 0),

        # Temporal focus
        'past_focus_score': counts.get('focuspast', 0),
        'present_focus_score': counts.get('focuspresent', 0),
        'future_focus_score': counts.get('focusfuture', 0),

        # Risk & Other
        'death_score': counts.get('death', 0),
        'swear_score': counts.get('swear', 0),
        'assent_score': counts.get('assent', 0),
        'nonfluency_score': counts.get('nonflu', 0),
    }

    # [FIX] Track source per dimension
    source_map = {
        'source_emotion': 'liwc',
        'source_cognitive': 'liwc',
        'source_pronouns': 'liwc',
        'source_temporal': 'liwc',
    }

    return overrides, source_map


def _compute_nrc_overrides(words_lower: List[str]) -> Tuple[Dict, Dict[str, str]]:
    """
    [FIX] Compute NRC features that OVERRIDE only the emotion dimension.
    All other dimensions remain from the regex base layer.
    Returns: (feature_overrides, source_map)
    """
    nrc = get_nrc()
    if nrc is None:
        return {}, {}

    nrc_counts = nrc.get_emotion_counts(words_lower)

    overrides = {
        # NRC maps to emotional dimensions only
        'anxiety_score': nrc_counts.get('fear', 0),
        'anger_score': nrc_counts.get('anger', 0),
        'sadness_score': nrc_counts.get('sadness', 0),
        'positive_score': (nrc_counts.get('joy', 0) +
                          nrc_counts.get('trust', 0) +
                          nrc_counts.get('anticipation', 0)),
        'negative_score': (nrc_counts.get('anger', 0) +
                          nrc_counts.get('disgust', 0) +
                          nrc_counts.get('fear', 0) +
                          nrc_counts.get('sadness', 0)),
    }

    # [FIX] NRC only covers emotion; other dimensions stay regex
    source_map = {
        'source_emotion': 'nrc',
        'source_cognitive': 'regex',
        'source_pronouns': 'regex',
        'source_temporal': 'regex',
    }

    return overrides, source_map


def extract_linguistic_features_post(text: str) -> Optional[Dict]:
    """
    Unified feature extraction with consistent source handling.

    [FIX] Architecture (v1.2):
      1. Always compute ALL regex features (complete base layer)
      2. If LIWC available → override regex for all LIWC-covered dimensions
      3. Else if NRC available → override regex for emotion dimensions only
      4. Regex base always provides behavioral supplements + cognitive/temporal
      5. VADER always computed

    This guarantees: every post gets a COMPLETE feature set regardless of
    which dictionary tier is active. No gaps when NRC is the emotion source.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return None

    # === PREPROCESSING ===
    clean = RE_URL.sub(' <URL> ', text)
    clean = RE_EMAIL.sub(' <EMAIL> ', clean)
    clean = clean.strip()

    if len(clean) < 10:
        return None

    words = RE_WORDS.findall(clean)
    words_lower = [w.lower() for w in words] #if len(w) > 1
    n_words = len(words_lower)

    if n_words < 3:
        return None

    n_chars = sum(len(w) for w in words_lower)
    sentences = [s.strip() for s in RE_SENTENCES.split(clean) if s.strip()]
    n_sentences = max(1, len(sentences))

    # === STEP 1: ALWAYS compute complete regex base ===
    features = _compute_regex_features(clean, words_lower, n_words, n_sentences)

    # Add structural & behavioral features (always from regex, never overridden)
    features.update({
        'n_words': n_words,
        'n_sentences': n_sentences,
        'n_chars': n_chars,
        'avg_word_len': n_chars / n_words,
        'avg_sent_len': n_words / n_sentences,
        'n_urls': len(RE_URL.findall(text)),
        'n_questions': len(RE_QUESTION.findall(clean)),
        'n_exclamations': len(RE_EXCLAMATION.findall(clean)),
        'n_quotations': len(RE_QUOTATION.findall(clean)),
        'n_ellipses': len(RE_ELLIPSIS.findall(clean)),
        'question_ratio': len(RE_QUESTION.findall(clean)) / n_sentences,
        'exclamation_ratio': len(RE_EXCLAMATION.findall(clean)) / n_sentences,
        # [FIX] Separated netspeak counts
        'netspeak_count': len(RE_NETSPEAK_WORDS.findall(clean)) + len(RE_NETSPEAK_EMOJI.findall(clean)),
        'filler_count': len(RE_FILLER.findall(clean)),

        # Source tracking defaults (will be overridden)
        'source_emotion': 'regex',
        'source_cognitive': 'regex',
        'source_pronouns': 'regex',
        'source_temporal': 'regex',

        # Placeholders for VADER & composites
        'vader_compound': 0,
        'vader_pos': 0,
        'vader_neg': 0,
        'vader_neu': 1.0,
        'vader_truncation_ratio': 0.0,
         'vader_compound_min': features.get('vader_compound', 0),
        'vader_compound_last': features.get('vader_compound', 0),
        'vader_compound_std': 0.0,
    
        'negative_net_score': 0,
        'self_focus_ratio': 0,
        'social_focus_ratio': 0,
        'cognitive_flexibility': 0.5,
        'temporal_balance': 0,
        'ari_score': 0,
        'mattr_score': calc_mattr(words_lower, window_size=min(50, n_words)),
    })

    # === STEP 2: Override with LIWC (if available) ===
    liwc_overrides, liwc_sources = _compute_liwc_overrides(words_lower)
    if liwc_overrides:
        features.update(liwc_overrides)
        features.update(liwc_sources)
    else:
        # === STEP 3: Override with NRC (if LIWC unavailable) ===
        nrc_overrides, nrc_sources = _compute_nrc_overrides(words_lower)
        if nrc_overrides:
            features.update(nrc_overrides)
            features.update(nrc_sources)
        # else: regex base remains (all features already set)

    # === STEP 4: VADER (Always) ===
    try:
        vader = get_vader()
        if vader:
            chunk_size = 2500  # 约等于 VADER 习惯处理的段落长度
            chunks = [clean[i:i+chunk_size] for i in range(0, len(clean), chunk_size)]
            
            chunk_compounds = []
            chunk_pos, chunk_neg, chunk_neu = [], [], []
            for chunk in chunks:
                vs = vader.polarity_scores(chunk)
                chunk_compounds.append(vs['compound'])
                chunk_pos.append(vs['pos'])
                chunk_neg.append(vs['neg'])
                chunk_neu.append(vs['neu'])
            
            if chunk_compounds:
                features.update({
                    'vader_compound': np.mean(chunk_compounds),        # 整体平均情感
                    'vader_pos': np.mean(chunk_pos),
                    'vader_neg': np.mean(chunk_neg),
                    'vader_neu': np.mean(chunk_neu),
                    'vader_truncation_ratio': 0.0,                     # 分段计算，无需截断
                    # [新增] 捕捉情感转折的关键特征
                    'vader_compound_min': np.min(chunk_compounds),     # 最负面瞬间（临床极高价值）
                    'vader_compound_last': chunk_compounds[-1],        # 最终情绪状态
                    'vader_compound_std': np.std(chunk_compounds),     # 情感波动性
                })
    except Exception:
        pass

    # === STEP 5: DERIVED COMPOSITE SCORES ===
    total_affect = features['positive_score'] + features['negative_score']
    if total_affect > 0:
        features['negative_net_score'] = (
            (features['positive_score'] - features['negative_score']) / total_affect
        )

    total_pronouns = (features['first_person_score'] + features['second_person_score'] +
                      features['third_person_score'] + features['impersonal_score'])
    if total_pronouns > 0:
        features['self_focus_ratio'] = features['first_person_score'] / total_pronouns
        features['social_focus_ratio'] = features['second_person_score'] / total_pronouns

    features['cognitive_flexibility'] = safe_divide(
        features['tentative_score'],
        features['tentative_score'] + features['certainty_score'] + 1e-10,
        default=0.5
    )

    time_total = features['past_focus_score'] + features['future_focus_score']
    if time_total > 0:
        features['temporal_balance'] = (
            (features['past_focus_score'] - features['future_focus_score']) / time_total
        )

    try:
        ari = 4.71 * features['avg_word_len'] + 0.5 * features['avg_sent_len'] - 21.43
        features['ari_score'] = max(-5, min(ari, 25))
    except Exception:
        features['ari_score'] = np.nan

    # Ensure all composite features exist
    required_composites = [
        'temporal_balance', 'negative_net_score', 'self_focus_ratio',
        'social_focus_ratio', 'cognitive_flexibility'
    ]
    for feat in required_composites:
        if feat not in features or features[feat] is None:
            features[feat] = np.nan

    return features


def extract_temporal_features_post(post_timestamp) -> Optional[Dict]:
    """Extract circadian and behavioral temporal features from a single timestamp"""
    try:
        ts = int(post_timestamp)
        if ts < 1000000000 or ts > 2000000000:
            return None

        hour_precise = (ts % 86400) / 3600.0
        day_of_week = datetime.utcfromtimestamp(ts).weekday()

        rad = hour_precise * (2 * pi / 24)
        sin_h = np.sin(rad)
        cos_h = np.cos(rad)

        is_night = 1 if (hour_precise >= 23 or hour_precise < 5) else 0
        is_weekend = 1 if day_of_week >= 5 else 0

        return {
            "hour_precise": hour_precise,
            "sin_hour": sin_h,
            "cos_hour": cos_h,
            "is_night": is_night,
            "is_weekend": is_weekend,
            "day_of_week": day_of_week,
            "timestamp": ts
        }
    except (ValueError, OverflowError, TypeError):
        return None


def _resolve_t0_date(user_df: pd.DataFrame) -> Optional[str]:
    """
    Resolve t0_date for a user, handling PENDING placeholders.
    
    Priority:
      1. Valid t0_date string (not PENDING/empty/nan)
      2. assigned_t0 Unix timestamp → convert to ISO date
      3. Median post_timestamp → convert to ISO date
      4. None (unresolvable)
    """
    # --- Priority 1: Valid t0_date ---
    if "t0_date" in user_df.columns:
        raw = user_df["t0_date"].iloc[0]
        if pd.notna(raw) and isinstance(raw, str) and raw.strip() not in (
            "", "PENDING", "nan", "None", "NaT", "NULL"
        ):
            # Validate it's actually parseable
            try:
                pd.to_datetime(raw)
                return raw
            except (ValueError, TypeError):
                pass  # Fall through to next priority

    # --- Priority 2: assigned_t0 (Unix timestamp) ---
    if "assigned_t0" in user_df.columns:
        assigned = user_df["assigned_t0"].iloc[0]
        if pd.notna(assigned):
            try:
                ts = int(assigned)
                if 1e9 < ts < 2e9:  # Seconds range (2001-2033)
                    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                elif 1e12 < ts < 2e12:  # Milliseconds range
                    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            except (ValueError, TypeError, OverflowError, OSError):
                pass

    # --- Priority 3: Median post timestamp ---
    if "post_timestamp" in user_df.columns:
        timestamps = user_df["post_timestamp"].dropna().values
        valid_ts = timestamps[(timestamps > 1e9) & (timestamps < 2e9)]
        if len(valid_ts) > 0:
            median_ts = int(np.median(valid_ts))
            try:
                return datetime.utcfromtimestamp(median_ts).strftime("%Y-%m-%d")
            except (OverflowError, OSError):
                pass

    return None
# ======================== USER AGGREGATION ========================

def aggregate_user_features(user_id: str, user_df: pd.DataFrame) -> Optional[Dict]:
    """
    Aggregate post-level to user-level with quality control.
    [FIX] Added: emotion inertia, post length CV, pronoun variability
    [FIX] Fixed: social jetlag with circular distance
    """
    n_posts = len(user_df)
    if n_posts < MIN_POSTS_FOR_FEATURES:
        return None

    # Metadata
    group = user_df["group"].iloc[0]
    disorder = user_df["disorder_group"].iloc[0] if "disorder_group" in user_df.columns else "unknown"
    # t0_date = user_df["t0_date"].iloc[0] if "t0_date" in user_df.columns else None
    t0_date = _resolve_t0_date(user_df)
    is_retrospective = user_df["is_retrospective"].iloc[0] if "is_retrospective" in user_df.columns else False

    # === 1. LINGUISTIC AGGREGATION ===
    # [FIX] Sort by timestamp for temporal sequence features (emotion inertia)
    if "post_timestamp" in user_df.columns:
        user_df = user_df.sort_values("post_timestamp").reset_index(drop=True)

    texts = user_df["clean_text"].fillna("").values
    post_features = []

    for t in texts:
        if isinstance(t, str) and len(t) > 10:
            pf = extract_linguistic_features_post(t)
            if pf:
                post_features.append(pf)

    total_words = sum(pf['n_words'] for pf in post_features)
    valid_posts = len(post_features)
    is_linguistic_reliable = (valid_posts >= 3 and total_words >= MIN_WORDS_FOR_LINGUISTIC)

    if is_linguistic_reliable:
        def sum_and_rate(key: str, divisor: int = total_words, multiplier: float = 100.0) -> float:
            return safe_divide(sum(pf[key] for pf in post_features), divisor, 0) * multiplier

        def mean_of(key: str) -> float:
            vals = [pf[key] for pf in post_features if not (isinstance(pf[key], float) and np.isnan(pf[key]))]
            return np.mean(vals) if vals else np.nan

        def std_of(key: str) -> float:
            vals = [pf[key] for pf in post_features if not (isinstance(pf[key], float) and np.isnan(pf[key]))]
            return np.std(vals) if len(vals) > 1 else 0.0

        affective_features = {
            'anxiety_rate': sum_and_rate('anxiety_score'),
            'anger_rate': sum_and_rate('anger_score'),
            'sadness_rate': sum_and_rate('sadness_score'),
            'positive_affect_rate': sum_and_rate('positive_score'),
            'negative_affect_rate': sum_and_rate('negative_score'),
            'net_emotion_tone': mean_of('negative_net_score'),
            'vader_compound_mean': mean_of('vader_compound'),
            'vader_compound_std': std_of('vader_compound'),
            'vader_positive_mean': mean_of('vader_pos'),
            'vader_negative_mean': mean_of('vader_neg'),
            'vader_within_post_volatility': mean_of('vader_compound_std'),   # 帖内波动
            'vader_compound_min_mean': mean_of('vader_compound_min'),        # 平均最负瞬间
            'vader_compound_last_mean': mean_of('vader_compound_last'),      # 平均终结情绪
            # 'emotional_volatility': std_of('vader_compound'),
        }

        pronoun_features = {
            'first_person_rate': sum_and_rate('first_person_score'),
            'second_person_rate': sum_and_rate('second_person_score'),
            'third_person_rate': sum_and_rate('third_person_score'),
            'impersonal_pronoun_rate': sum_and_rate('impersonal_score'),
            'self_focus_index': mean_of('self_focus_ratio'),
            'social_focus_index': mean_of('social_focus_ratio'),
            # [NEW] Pronoun variability across posts
            'self_focus_variability': std_of('self_focus_ratio'),
            'social_focus_variability': std_of('social_focus_ratio'),
        }

        cognitive_features = {
            'insight_rate': sum_and_rate('insight_score'),
            'causal_thinking_rate': sum_and_rate('causal_score'),
            'certainty_rate': sum_and_rate('certainty_score'),
            'tentative_language_rate': sum_and_rate('tentative_score'),
            'discrepancy_rate': sum_and_rate('discrepancy_score'),
            'inhibition_rate': sum_and_rate('inhibition_score'),
            'cognitive_flexibility': mean_of('cognitive_flexibility'),
        }

        temporal_ling_features = {
            'past_focus_rate': sum_and_rate('past_focus_score'),
            'present_focus_rate': sum_and_rate('present_focus_score'),
            'future_focus_rate': sum_and_rate('future_focus_score'),
            'temporal_balance': mean_of('temporal_balance'),
        }

        risk_features = {
            'death_mortality_rate': sum_and_rate('death_score'),
            'swearing_rate': sum_and_rate('swear_score'),
            'assent_rate': sum_and_rate('assent_score'),
            'nonfluency_rate': sum_and_rate('nonfluency_score'),
            'inhibition_marker_rate': sum_and_rate('inhibition_score'),
        }

        behavioral_features = {
            'question_ratio': mean_of('question_ratio'),
            'exclamation_ratio': mean_of('exclamation_ratio'),
            'url_usage_rate': sum_and_rate('n_urls'),
            'netspeak_usage_rate': sum_and_rate('netspeak_count'),
            'filler_word_rate': sum_and_rate('filler_count'),
        }

        readability_features = {
            'ari_mean': mean_of('ari_score'),
            'ari_std': std_of('ari_score'),
            'mattr_vocabulary_diversity': mean_of('mattr_score'),
        }

        # [NEW] Post length variability
        word_counts = [pf['n_words'] for pf in post_features]
        word_count_mean = np.mean(word_counts)
        word_count_std = np.std(word_counts) if len(word_counts) > 1 else 0.0
        post_length_features = {
            'post_length_mean': float(word_count_mean),
            'post_length_std': float(word_count_std),
            'post_length_cv': safe_divide(word_count_std, word_count_mean, 0),
        }

        # [NEW] Emotion inertia: autocorrelation of VADER compound across posts
        vader_series = [pf['vader_compound'] for pf in post_features]
        if len(vader_series) > 3:
            vader_arr = np.array(vader_series, dtype=float)
            vader_mean = np.mean(vader_arr)
            vader_var = np.var(vader_arr)
            if vader_var > 1e-10:
                autocorr = np.mean((vader_arr[:-1] - vader_mean) * (vader_arr[1:] - vader_mean)) / vader_var
                emotion_inertia = float(np.clip(autocorr, -1, 1))
            else:
                emotion_inertia = 0.0
        else:
            emotion_inertia = np.nan
        temporal_dynamic_features = {
            'emotion_inertia': emotion_inertia,
        }

        # [NEW] VADER truncation statistics
        vader_truncation_features = {
            'vader_truncation_rate': np.mean([pf['vader_truncation_ratio'] for pf in post_features]),
        }

        source_flags = {
            'pct_liwc_source': sum(1 for pf in post_features if pf['source_emotion'] == 'liwc') / valid_posts,
            'pct_nrc_source': sum(1 for pf in post_features if pf['source_emotion'] == 'nrc') / valid_posts,
            'pct_regex_source': sum(1 for pf in post_features if pf['source_emotion'] == 'regex') / valid_posts,
        }

        ling_agg = {}
        for d in [affective_features, pronoun_features, cognitive_features,
                  temporal_ling_features, risk_features, behavioral_features,
                  readability_features, post_length_features, temporal_dynamic_features,
                  vader_truncation_features, source_flags]:
            ling_agg.update(d)
    else:
        ling_agg = {
            'anxiety_rate': np.nan, 'anger_rate': np.nan, 'sadness_rate': np.nan,
            'positive_affect_rate': np.nan, 'negative_affect_rate': np.nan,
            'net_emotion_tone': np.nan, 'vader_compound_mean': np.nan,
            'vader_compound_std': np.nan, 
            'first_person_rate': np.nan, 'second_person_rate': np.nan,
            'self_focus_index': np.nan, 'insight_rate': np.nan,
            'tentative_language_rate': np.nan, 'death_mortality_rate': np.nan,
            'temporal_balance': np.nan, 'mattr_vocabulary_diversity': np.nan,
            'self_focus_variability': np.nan, 'social_focus_variability': np.nan,
            'post_length_cv': np.nan, 'emotion_inertia': np.nan,
            'vader_truncation_rate': 0.0,
            'pct_liwc_source': 0.0, 'pct_nrc_source': 0.0, 'pct_regex_source': 1.0
        }

    # === 2. TEMPORAL-BEHAVIORAL AGGREGATION ===
    timestamps = user_df["post_timestamp"].dropna().astype(np.int64).values
    valid_ts = timestamps[(timestamps > 1000000000) & (timestamps < 2000000000)]
    # [FIX] Use MIN_POSTS_FOR_TEMPORAL (15) instead of MIN_POSTS_FOR_FEATURES (10)
    is_temporal_reliable = len(valid_ts) >= MIN_POSTS_FOR_TEMPORAL

    if is_temporal_reliable:
        n = len(valid_ts)
        min_ts, max_ts = valid_ts.min(), valid_ts.max()
        tenure_days = (max_ts - min_ts) / 86400.0
        posts_per_day = n / max(1.0, tenure_days)

        # Circadian statistics (circular mean)
        hours = (valid_ts % 86400) / 3600.0
        rads = hours * (2 * pi / 24)
        sin_mean = np.mean(np.sin(rads))
        cos_mean = np.mean(np.cos(rads))

        mean_angle = np.arctan2(sin_mean, cos_mean)
        circadian_mean_hour = mean_angle * (24 / (2 * pi))
        if circadian_mean_hour < 0:
            circadian_mean_hour += 24.0
        circadian_r = np.sqrt(sin_mean**2 + cos_mean**2)

        night_ratio = np.mean((hours >= 23) | (hours < 5))
        days_of_week = np.array([datetime.utcfromtimestamp(int(ts)).weekday() for ts in valid_ts])
        weekend_ratio = np.mean(days_of_week >= 5)

        # [FIX] Social jetlag with circular distance
        weekday_mask = days_of_week < 5
        if np.sum(weekday_mask) > 0 and np.sum(~weekday_mask) > 0:
            # Use circular mean for each group
            wkd_rads = hours[weekday_mask] * (2 * pi / 24)
            wke_rads = hours[~weekday_mask] * (2 * pi / 24)

            wkd_sin, wkd_cos = np.mean(np.sin(wkd_rads)), np.mean(np.cos(wkd_rads))
            wke_sin, wke_cos = np.mean(np.sin(wke_rads)), np.mean(np.cos(wke_rads))

            wkd_mean_hour = np.arctan2(wkd_sin, wkd_cos) * (24 / (2 * pi))
            if wkd_mean_hour < 0:
                wkd_mean_hour += 24.0
            wke_mean_hour = np.arctan2(wke_sin, wke_cos) * (24 / (2 * pi))
            if wke_mean_hour < 0:
                wke_mean_hour += 24.0

            social_jetlag = signed_circular_diff(wke_mean_hour, wkd_mean_hour)
            has_mixed_activity = 1
        else:
            social_jetlag = np.nan
            has_mixed_activity = 0

        # Posting intervals
        sorted_ts = np.sort(valid_ts)
        intervals = np.diff(sorted_ts).astype(np.float64)
        intervals = intervals[intervals > 0]

        if len(intervals) > 1:
            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            interval_median = np.median(intervals)
            interval_cv = interval_std / interval_mean
            burstiness = (interval_std - interval_mean) / (interval_std + interval_mean + 1e-10)
            interval_kurt = kurtosis(intervals, fisher=True)
            interval_median_hours = interval_median / 3600.0
            interval_mean_hours = interval_mean / 3600.0
        else:
            interval_cv = np.nan
            interval_median_hours = np.nan
            burstiness = np.nan
            interval_kurt = np.nan
            interval_mean_hours = np.nan

        # Active days
        active_dates = {int(ts) // 86400 for ts in valid_ts}
        n_active_days = len(active_dates)
        active_day_ratio = n_active_days / max(1, int(tenure_days) + 1) if tenure_days > 0 else 1.0

        sorted_days = sorted(active_dates)
        if len(sorted_days) > 1:
            day_gaps = np.diff(sorted_days)
            max_gap_days = int(np.max(day_gaps))
            mean_gap_days = float(np.mean(day_gaps))
            gap_cv = np.std(day_gaps) / np.mean(day_gaps) if len(day_gaps) > 1 else 0.0
        else:
            max_gap_days = 0
            mean_gap_days = 0.0
            gap_cv = 0.0

        # Entropy measures
        hour_bins, _ = np.histogram(hours, bins=24, range=(0, 24))
        hour_probs = hour_bins / hour_bins.sum() if hour_bins.sum() > 0 else np.ones(24) / 24
        hourly_entropy = entropy(hour_probs + 1e-10)

        dow_bins, _ = np.histogram(days_of_week, bins=7, range=(0, 7))
        dow_probs = dow_bins / dow_bins.sum() if dow_bins.sum() > 0 else np.ones(7) / 7
        weekly_entropy = entropy(dow_probs + 1e-10)

        # Sleep disruption
        sleep_hours_mask = (hours >= 0) & (hours < 6)
        sleep_disruption_index = np.mean(sleep_hours_mask)

        temporal_agg = {
            'n_posts_total': n_posts,
            'tenure_days': tenure_days,
            'posts_per_day': posts_per_day,
            'posts_per_active_day': n / max(1, n_active_days),
            'circadian_mean_hour': circadian_mean_hour,
            'circadian_concentration': circadian_r,
            'night_activity_ratio': night_ratio,
            'weekend_activity_ratio': weekend_ratio,
            'social_jetlag_hours': social_jetlag,
            'has_weekday_weekend_mixed': has_mixed_activity,
            'sleep_disruption_index': sleep_disruption_index,
            'posting_regularity_cv': interval_cv,
            'posting_burstiness': burstiness,
            'interval_median_hours': interval_median_hours,
            'interval_mean_hours': interval_mean_hours,
            'interval_kurtosis': interval_kurt,
            'n_active_days': n_active_days,
            'active_day_ratio': active_day_ratio,
            'max_gap_days': max_gap_days,
            'mean_gap_days': mean_gap_days,
            'gap_regularity_cv': gap_cv,
            'hourly_entropy': hourly_entropy,
            'weekly_entropy': weekly_entropy,
            'temporal_diversity_index': (hourly_entropy + weekly_entropy) / 2,
        }
    else:
        temporal_agg = {
            'n_posts_total': n_posts,
            'tenure_days': np.nan, 'posts_per_day': np.nan, 'posts_per_active_day': np.nan,
            'circadian_mean_hour': np.nan, 'circadian_concentration': np.nan,
            'night_activity_ratio': np.nan, 'weekend_activity_ratio': np.nan,
            'social_jetlag_hours': np.nan, 'sleep_disruption_index': np.nan,
            'posting_regularity_cv': np.nan, 'posting_burstiness': np.nan,
            'interval_median_hours': np.nan, 'interval_mean_hours': np.nan,
            'interval_kurtosis': np.nan, 'n_active_days': np.nan,
            'active_day_ratio': np.nan, 'max_gap_days': np.nan,
            'mean_gap_days': np.nan, 'gap_regularity_cv': np.nan,
            'hourly_entropy': np.nan, 'weekly_entropy': np.nan,
            'temporal_diversity_index': np.nan,
        }

    # === 3. SOCIAL INTERACTION FEATURES ===
    scores = user_df["score"].fillna(0).astype(float).values

    if "data_type" in user_df.columns:
        data_types = user_df["data_type"].values
        submission_mask = (data_types == "submission")
    else:
        submission_mask = np.zeros(n_posts, dtype=bool)

    n_submissions = int(np.sum(submission_mask))
    n_comments = n_posts - n_submissions

    if len(scores) > 0:
        score_mean = float(np.mean(scores))
        score_median = float(np.median(scores))
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        score_max = float(np.max(scores))
        score_min = float(np.min(scores))
        neg_score_ratio = float(np.mean(scores < 0))
        high_engagement_ratio = (
            float(np.mean(scores > np.percentile(scores, 90)))
            if len(scores) > 10 else 0.0
        )
    else:
        score_mean = score_median = score_std = score_max = score_min = np.nan
        neg_score_ratio = high_engagement_ratio = np.nan

    if "is_top_level" in user_df.columns and n_comments > 0:
        comment_mask = ~submission_mask
        if np.any(comment_mask):
            top_level_vals = user_df.loc[comment_mask, "is_top_level"].fillna(False).astype(bool)
            top_level_ratio = float(np.mean(top_level_vals))
            avg_comment_depth = 1.0 if top_level_ratio > 0.5 else 2.0
        else:
            top_level_ratio = np.nan
            avg_comment_depth = np.nan
    else:
        top_level_ratio = np.nan
        avg_comment_depth = np.nan

    if "num_comments" in user_df.columns and n_submissions > 0:
        submission_indices = submission_mask
        if np.any(submission_indices):
            comments_received = user_df.loc[submission_indices, "num_comments"].fillna(0).astype(float)
            avg_comments_received = float(np.mean(comments_received))
            total_comments_received = int(np.sum(comments_received))
            discussion_initiated_ratio = float(np.mean(comments_received > 0))
        else:
            avg_comments_received = total_comments_received = discussion_initiated_ratio = np.nan
    else:
        avg_comments_received = total_comments_received = discussion_initiated_ratio = np.nan

    controversy_rate = np.nan
    if "controversiality" in user_df.columns:
        controversy_vals = user_df["controversiality"].fillna(0).values
        controversy_rate = float(np.mean(controversy_vals > 0)) if len(controversy_vals) > 0 else 0.0

    gilded_rate = np.nan
    if "gilded" in user_df.columns:
        gilded_count = user_df["gilded"].fillna(0).astype(int).sum()
        gilded_rate = gilded_count / n_posts

    self_post_ratio = np.nan
    if n_submissions > 0 and "is_self" in user_df.columns:
        self_vals = user_df.loc[submission_mask, "is_self"].fillna(False).astype(bool)
        self_post_ratio = float(np.mean(self_vals))

    social_agg = {
        'submission_ratio': safe_divide(n_submissions, n_posts, 0),
        'n_submissions': n_submissions,
        'n_comments': n_comments,
        'top_level_comment_ratio': top_level_ratio,
        'avg_comment_depth': avg_comment_depth,
        'score_mean': score_mean,
        'score_median': score_median,
        'score_std': score_std,
        'score_max': score_max,
        'score_min': score_min,
        'score_range': score_max - score_min if not np.isnan(score_max) else np.nan,
        'neg_score_ratio': neg_score_ratio,
        'high_engagement_ratio': high_engagement_ratio,
        'avg_comments_received': avg_comments_received,
        'total_comments_received': total_comments_received,
        'discussion_initiated_ratio': discussion_initiated_ratio,
        'self_post_ratio': self_post_ratio,
        'controversy_rate': controversy_rate,
        'gilded_rate': gilded_rate,
    }

    # === 4. COMMUNITY & CONTEXT FEATURES ===
    if "subreddit" in user_df.columns:
        subreddits = user_df["subreddit"].fillna("unknown").str.lower().values
        sub_counter = Counter(subreddits)
        total_sub_posts = sum(sub_counter.values())

        if total_sub_posts > 0:
            sub_probs = np.array(list(sub_counter.values())) / total_sub_posts
            subreddit_entropy = float(entropy(sub_probs + 1e-10))
            subreddit_herfindahl = float(np.sum(sub_probs ** 2))

            mh_posts = sum(1 for s in subreddits if s in MENTAL_HEALTH_SUBREDDITS)
            mh_ratio = safe_divide(mh_posts, n_posts, 0)

            category_counts = defaultdict(int)
            for s in subreddits:
                assigned = False
                for cat, subs_set in SUBREDDIT_CATEGORIES.items():
                    if s in subs_set:
                        category_counts[cat] += 1
                        assigned = True
                        break
                if not assigned:
                    category_counts['other'] += 1

            unique_subs = len(sub_counter)
            primary_subreddit = sub_counter.most_common(1)[0][0] if sub_counter else "unknown"
            primary_subreddit_ratio = sub_counter.most_common(1)[0][1] / n_posts if sub_counter else 0

            community_agg = {
                'n_unique_subreddits': unique_subs,
                'subreddit_entropy': subreddit_entropy,
                'subreddit_herfindahl': subreddit_herfindahl,
                'primary_subreddit': primary_subreddit,
                'primary_subreddit_ratio': primary_subreddit_ratio,
                'mental_health_sub_ratio': mh_ratio,
                'support_sub_ratio': safe_divide(category_counts.get('support', 0), n_posts, 0),
                'entertainment_sub_ratio': safe_divide(category_counts.get('entertainment', 0), n_posts, 0),
                'information_sub_ratio': safe_divide(category_counts.get('information', 0), n_posts, 0),
                'social_sub_ratio': safe_divide(category_counts.get('social', 0), n_posts, 0),
                'other_sub_ratio': safe_divide(category_counts.get('other', 0), n_posts, 0),
            }
        else:
            community_agg = _nan_community_agg()
    else:
        community_agg = _nan_community_agg()

    # === 5. COMBINE ALL FEATURES ===
    result = {
        'user_id_hash': user_id,
         'group': 'control' if group.startswith('control') else group,
        'disorder_group': disorder if group == 'patient' else 'control',
        'is_retrospective': bool(is_retrospective),
        # 't0_date': str(t0_date) if t0_date else None,
        't0_date': t0_date,
        'analysis_timestamp': datetime.now().isoformat(),
        'quality_n_posts_raw': n_posts,
        'quality_total_words': total_words,
        'quality_n_posts_valid': valid_posts,
        'quality_is_linguistic_reliable': is_linguistic_reliable,
        'quality_is_temporal_reliable': is_temporal_reliable,
        'quality_linguistic_coverage': safe_divide(total_words, n_posts, 0),
        'quality_temporal_span_days': temporal_agg.get('tenure_days', np.nan),
        'quality_source_liwc_pct': ling_agg.get('pct_liwc_source', 0.0),
        'quality_source_nrc_pct': ling_agg.get('pct_nrc_source', 0.0),
        'quality_source_regex_pct': ling_agg.get('pct_regex_source', 0.0),
    }

    result.update(ling_agg)
    result.update(temporal_agg)
    result.update(social_agg)
    result.update(community_agg)

    return result


def _nan_community_agg() -> Dict:
    """Helper: return NaN-filled community features"""
    keys = [
        'n_unique_subreddits', 'subreddit_entropy', 'subreddit_herfindahl',
        'primary_subreddit', 'primary_subreddit_ratio', 'mental_health_sub_ratio',
        'support_sub_ratio', 'entertainment_sub_ratio', 'information_sub_ratio',
        'social_sub_ratio', 'other_sub_ratio'
    ]
    return {k: np.nan for k in keys}


# ======================== PROCESSING & I/O ========================

def get_partition_dirs(silver_dir: str) -> List[str]:
    """Discover partition directories"""
    if not os.path.exists(silver_dir):
        logger.error(f"Directory not found: {silver_dir}")
        return []

    dirs = []
    for entry in os.listdir(silver_dir):
        full_path = os.path.join(silver_dir, entry)
        if os.path.isdir(full_path) and entry.startswith('part_'):
            dirs.append(full_path)

    return sorted(dirs)


def process_partition(partition_path: str) -> List[Dict]:
    """
    Process single partition with memory-efficient streaming.
    [FIX] Reduced memory footprint: process users incrementally.
    """
    logger.info(f"Processing {os.path.basename(partition_path)}")

    init_worker()

    status = _dict_manager.get_status()
    logger.info(f"Worker PID {status['pid']}: LIWC={status['liwc_available']}, "
                f"NRC={status['nrc_available']}, VADER={status['vader_available']}, "
                f"Source={status['effective_source']}")

    parquet_files = [f for f in os.listdir(partition_path) if f.endswith('.parquet')]
    if not parquet_files:
        logger.warning(f"No parquet files in {partition_path}")
        return []

    # [FIX] Two-pass approach to reduce peak memory:
    # Pass 1: Collect user IDs and their file/row locations
    # Pass 2: Process each user individually
    # However, for simplicity and compatibility, we use a single-pass
    # approach with incremental processing and explicit cleanup.

    user_data = defaultdict(list)

    for pf_name in parquet_files:
        pf_path = os.path.join(partition_path, pf_name)
        try:
            pf = pq.ParquetFile(pf_path)

            for batch in pf.iter_batches(batch_size=min(50000, BATCH_SIZE)):
                df = batch.to_pandas()

                for uid, udf in df.groupby('user_id_hash'):
                    user_data[uid].append(udf)

                del df
                gc.collect()

        except Exception as e:
            logger.error(f"Error reading {pf_name}: {e}")
            continue

    # [FIX] Process users and release memory incrementally
    results = []
    processed_uids = []

    for uid, dfs in tqdm(user_data.items(), desc=f"Aggregating {os.path.basename(partition_path)}"):
        try:
            user_df = pd.concat(dfs, ignore_index=True)
            feats = aggregate_user_features(uid, user_df)
            if feats:
                results.append(feats)
            del user_df
        except Exception as e:
            logger.error(f"User {uid[:8]} aggregation failed: {e}")

        # [FIX] Free individual user DataFrames after processing
        processed_uids.append(uid)

    # [FIX] Explicitly free memory
    for uid in processed_uids:
        if uid in user_data:
            del user_data[uid]
    del user_data
    gc.collect()

    return results


def load_psm_weights() -> Optional[pd.DataFrame]:
    """Load propensity score weights if available"""
    if not os.path.exists(PSM_MATCHED_PATH):
        logger.info("PSM weights not found, continuing without")
        return None

    try:
        df = pd.read_parquet(PSM_MATCHED_PATH)
        weight_cols = ['user_id_hash', 'iptw_att_weight', 'ps_score', 'eb_weight']
        available = [c for c in weight_cols if c in df.columns]

        if len(available) > 1:
            logger.info(f"Loaded PSM weights: {len(available)-1} weight types")
            return df[available]
        return None
    except Exception as e:
        logger.error(f"PSM load error: {e}")
        return None


def build_feature_metadata(features_df: pd.DataFrame) -> Dict:
    """
    [FIX] Build structured feature metadata for Step 02 compatibility.
    Maps each feature to its dimension and source type.
    """
    affective_keys = [
    'anxiety_rate', 'anger_rate', 'sadness_rate',
    'positive_affect_rate', 'negative_affect_rate',
    'net_emotion_tone', 'vader_compound_mean', 'vader_compound_std',
    'vader_positive_mean', 'vader_negative_mean',
    'vader_within_post_volatility',     # ← 新增
    'vader_compound_min_mean',          # ← 新增
    'vader_compound_last_mean',         # ← 新增
    'vader_truncation_rate',            # ← 新增
    'emotion_inertia',
    ]
    pronoun_keys = [
        'first_person_rate', 'second_person_rate', 'third_person_rate',
        'impersonal_pronoun_rate', 'self_focus_index', 'social_focus_index',
        'self_focus_variability', 'social_focus_variability',
    ]
    cognitive_keys = [
        'insight_rate', 'causal_thinking_rate', 'certainty_rate',
        'tentative_language_rate', 'discrepancy_rate', 'inhibition_rate',
        'cognitive_flexibility',
    ]
    temporal_ling_keys = [
        'past_focus_rate', 'present_focus_rate', 'future_focus_rate',
        'temporal_balance',
    ]
    risk_keys = [
        'death_mortality_rate', 'swearing_rate', 'assent_rate',
        'nonfluency_rate', 'inhibition_marker_rate',
    ]
    behavioral_keys = [
        'question_ratio', 'exclamation_ratio', 'url_usage_rate',
        'netspeak_usage_rate', 'filler_word_rate',
    ]
    readability_keys = [
        'ari_mean', 'ari_std', 'mattr_vocabulary_diversity',
    ]
    post_length_keys = [
        'post_length_mean', 'post_length_std', 'post_length_cv',
    ]
    temporal_keys = [
        'n_posts_total', 'tenure_days', 'posts_per_day', 'posts_per_active_day',
        'circadian_mean_hour', 'circadian_concentration',
        'night_activity_ratio', 'weekend_activity_ratio',
        'social_jetlag_hours', 'sleep_disruption_index',
        'posting_regularity_cv', 'posting_burstiness',
        'interval_median_hours', 'interval_mean_hours', 'interval_kurtosis',
        'n_active_days', 'active_day_ratio',
        'max_gap_days', 'mean_gap_days', 'gap_regularity_cv',
        'hourly_entropy', 'weekly_entropy', 'temporal_diversity_index',
    ]
    social_keys = [
        'submission_ratio', 'n_submissions', 'n_comments',
        'top_level_comment_ratio', 'avg_comment_depth',
        'score_mean', 'score_median', 'score_std', 'score_max', 'score_min', 'score_range',
        'neg_score_ratio', 'high_engagement_ratio',
        'avg_comments_received', 'total_comments_received',
        'discussion_initiated_ratio', 'self_post_ratio',
        'controversy_rate', 'gilded_rate',
    ]
    community_keys = [
        'n_unique_subreddits', 'subreddit_entropy', 'subreddit_herfindahl',
        'primary_subreddit_ratio', 'mental_health_sub_ratio',
        'support_sub_ratio', 'entertainment_sub_ratio',
        'information_sub_ratio', 'social_sub_ratio', 'other_sub_ratio',
    ]
    quality_keys = [c for c in features_df.columns if c.startswith('quality_')]

    # Only include columns that actually exist in the DataFrame
    col_set = set(features_df.columns)

    metadata = {}
    for dim_name, keys in [
        ('affective', affective_keys),
        ('pronoun', pronoun_keys),
        ('cognitive', cognitive_keys),
        ('temporal_linguistic', temporal_ling_keys),
        ('risk', risk_keys),
        ('behavioral', behavioral_keys),
        ('readability', readability_keys),
        ('post_length', post_length_keys),
        ('temporal', temporal_keys),
        ('social', social_keys),
        ('community', community_keys),
        ('quality', quality_keys),
    ]:
        existing = [k for k in keys if k in col_set]
        if existing:
            metadata[dim_name] = existing

    # Add log-transformed features
    log_cols = [c for c in features_df.columns if c.endswith('_log')]
    if log_cols:
        metadata['log_transforms'] = log_cols

    # Add signed log features
    signed_log_cols = [c for c in features_df.columns if c.endswith('_log_signed')]
    if signed_log_cols:
        metadata['signed_log_transforms'] = signed_log_cols

    return metadata


def run_extraction_pipeline():
    """Main execution pipeline"""
    logger.info("=" * 80)
    logger.info("MENTAL HEALTH FEATURE EXTRACTION PIPELINE (Production v1.2)")
    logger.info("Fixes: present_focus, unified source, NRC reverse index,")
    logger.info("       circular jetlag, memory optimization, emotion inertia")
    logger.info("=" * 80)

    # Validate environment
    liwc_available = USE_LIWC and os.path.exists(LIWC_DICTIONARY_PATH)
    nrc_available = os.path.exists(NRC_LEXICON_PATH)

    if USE_LIWC and not liwc_available:
        logger.error(f"LIWC enabled but not found: {LIWC_DICTIONARY_PATH}")
        logger.info("Set USE_LIWC=False to use NRC/Regex fallback")
        if SOURCE_POLICY == "strict":
            logger.error("STRICT mode: Aborting due to LIWC unavailability")
            return None

    if not nrc_available:
        logger.warning(f"NRC lexicon not found: {NRC_LEXICON_PATH}")

    # Test initialization in main process
    logger.info("Testing dictionary initialization in main process...")
    init_worker()
    main_status = _dict_manager.get_status()
    logger.info(f"Main process: LIWC={main_status['liwc_available']}, "
                f"NRC={main_status['nrc_available']}, VADER={main_status['vader_available']}, "
                f"Effective source={main_status['effective_source']}")

    # Get partitions
    partitions = get_partition_dirs(SILVER_DATA_DIR)
    if not partitions:
        logger.error(f"No partitions found in {SILVER_DATA_DIR}")
        return None

    logger.info(f"Found {len(partitions)} partitions to process")

    # Process with multiprocessing
    all_results = []

    if NUM_CORES > 1 and len(partitions) > 1:
        logger.info(f"Parallel processing with {NUM_CORES} cores")

        # [FIX] Process in smaller chunks to limit memory
        chunk_size = min(NUM_CORES, len(partitions))
        for i in range(0, len(partitions), chunk_size):
            chunk = partitions[i:i + chunk_size]

            with Pool(processes=min(NUM_CORES, len(chunk)), initializer=init_worker) as pool:
                chunk_results = list(tqdm(
                    pool.imap_unordered(process_partition, chunk),
                    total=len(chunk),
                    desc=f"Batch {i // chunk_size + 1}"
                ))

            for res in chunk_results:
                all_results.extend(res)

            logger.info(f"Progress: {len(all_results)} users aggregated")
            gc.collect()
    else:
        logger.info("Sequential processing")
        for partition in tqdm(partitions, desc="Partitions"):
            results = process_partition(partition)
            all_results.extend(results)
            gc.collect()

    if not all_results:
        logger.error("No users successfully processed")
        return None

    # Create DataFrame
    logger.info("Compiling features...")
    features_df = pd.DataFrame(all_results)
    del all_results
    gc.collect()

    # Remove duplicates
    n_before = len(features_df)
    features_df = features_df.drop_duplicates(subset=['user_id_hash'])
    if len(features_df) < n_before:
        logger.warning(f"Removed {n_before - len(features_df)} duplicate users")

    # Merge PSM weights
    psm_df = load_psm_weights()
    if psm_df is not None:
        features_df = features_df.merge(psm_df, on='user_id_hash', how='left')
        logger.info("Merged propensity score weights")

    # Scientific feature engineering transformations (log transforms only)
    logger.info("Applying scientific transformations...")

    skewed_temporal = [
        'posts_per_day', 'posts_per_active_day', 'interval_median_hours',
        'max_gap_days', 'tenure_days', 'n_posts_total'
    ]

    for col in skewed_temporal:
        if col in features_df.columns:
            features_df[f'{col}_log'] = np.log1p(features_df[col].fillna(0).clip(lower=0))

    if 'score_mean' in features_df.columns:
        features_df['score_mean_log_signed'] = (
            np.sign(features_df['score_mean']) *
            np.log1p(np.abs(features_df['score_mean']))
        )

    # [FIX] Removed Z-score generation — this belongs in Step 02 after proper
    # train/test splitting to avoid data leakage. The previous implementation
    # computed z-scores using the entire dataset before splitting.

    # Build and save feature metadata
    logger.info("Building feature metadata...")
    feature_meta = build_feature_metadata(features_df)

    # Save outputs
    # [FIX] Use consistent file naming for Step 02 compatibility
    output_path = os.path.join(FEATURES_DIR, RAW_FEATURES_FILE)
    features_df.to_parquet(output_path, compression='zstd', index=False)
    logger.info(f"Saved features to: {output_path}")
    logger.info(f"Shape: {features_df.shape}")

    # Save feature metadata
    metadata_path = os.path.join(FEATURES_DIR, FEATURE_METADATA_FILE)
    with open(metadata_path, 'w') as f:
        json.dump(feature_meta, f, indent=2)
    logger.info(f"Saved feature metadata to: {metadata_path}")

    # Group statistics
    if 'group' in features_df.columns:
        logger.info(f"Group distribution:\n{features_df['group'].value_counts()}")
    if 'disorder_group' in features_df.columns:
        logger.info(f"Disorder distribution:\n{features_df['disorder_group'].value_counts()}")

    # Source coverage statistics
    logger.info("Dictionary source coverage:")
    for source in ['quality_source_liwc_pct', 'quality_source_nrc_pct', 'quality_source_regex_pct']:
        if source in features_df.columns:
            mean_pct = features_df[source].mean() * 100
            logger.info(f"  {source}: {mean_pct:.1f}% average coverage")

    metadata_cols = {'user_id_hash', 'group', 'disorder_group', 'is_retrospective',
                 't0_date', 'analysis_timestamp', 'primary_subreddit'}
    # Quality report
    quality_summary = {
        'total_users': len(features_df),
        'patients': int((features_df['group'] == 'patient').sum()) if 'group' in features_df.columns else 0,
        'controls': int((features_df['group'] == 'control').sum()) if 'group' in features_df.columns else 0,
        'linguistically_reliable': int(features_df['quality_is_linguistic_reliable'].sum())
            if 'quality_is_linguistic_reliable' in features_df.columns else 0,
        'temporally_reliable': int(features_df['quality_is_temporal_reliable'].sum())
            if 'quality_is_temporal_reliable' in features_df.columns else 0,
        'liwc_coverage_mean': float(features_df['quality_source_liwc_pct'].mean())
            if 'quality_source_liwc_pct' in features_df.columns else 0.0,
        'nrc_coverage_mean': float(features_df['quality_source_nrc_pct'].mean())
            if 'quality_source_nrc_pct' in features_df.columns else 0.0,
        'regex_coverage_mean': float(features_df['quality_source_regex_pct'].mean())
            if 'quality_source_regex_pct' in features_df.columns else 0.0,
        'n_features': len([c for c in features_df.columns if c not in metadata_cols]),
        'effective_source': get_effective_source(),
        'pipeline_version': 'v1.2',
        'extraction_timestamp': datetime.now().isoformat(),
        'feature_dimensions': {k: len(v) for k, v in feature_meta.items()},
    }

    quality_path = os.path.join(FEATURES_DIR, QUALITY_REPORT_FILE)
    with open(quality_path, 'w') as f:
        json.dump(quality_summary, f, indent=2)

    logger.info("Quality Report:")
    for k, v in quality_summary.items():
        logger.info(f"  {k}: {v}")

    # [FIX] Also save timestamped versions for audit trail
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = os.path.join(FEATURES_DIR, f"user_features_{timestamp}.parquet")
    features_df.to_parquet(audit_path, compression='zstd', index=False)
    logger.info(f"Audit copy saved to: {audit_path}")

    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

    return features_df


if __name__ == "__main__":
    run_extraction_pipeline()