"""
utils.py — Shared utility functions for Financial News Emotion Intelligence
============================================================
Provides:
  - Config loading
  - Logging setup
  - Date utilities
  - Text utilities
  - File I/O helpers
  - Emotion score normalization
"""

import os
import re
import json
import logging
import hashlib
import unicodedata
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Union

import yaml
import pandas as pd
import numpy as np


# ==============================================================
# 1. CONFIG
# ==============================================================

def load_config(config_path: Optional[str] = None) -> dict:
    """Load master config.yaml. Auto-discovers if not specified."""
    if config_path is None:
        # Walk up directories to find config.yaml
        current = Path(__file__).resolve().parent
        for _ in range(5):
            candidate = current / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent
    if config_path is None:
        raise FileNotFoundError("config.yaml not found. Run from project root.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================
# 2. LOGGING
# ==============================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger(__name__)


# ==============================================================
# 3. PATH HELPERS
# ==============================================================

def get_project_root() -> Path:
    """Return absolute path to project root (where config.yaml lives)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Project root (config.yaml) not found.")


def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ==============================================================
# 4. TEXT UTILITIES
# ==============================================================

def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    # Remove script/style blocks first
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#\d+;', ' ', text)
    return text


def clean_urls(text: str) -> str:
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    return text


def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII-compatible form."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """Remove non-alphanumeric characters. Optionally keep key punctuation."""
    if not isinstance(text, str):
        return ""
    if keep_punctuation:
        # Keep letters, numbers, spaces, and core punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"\-\%\$]', ' ', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text


def basic_clean(text: str) -> str:
    """Full cleaning pipeline for classical NLP."""
    text = clean_html(text)
    text = clean_urls(text)
    text = normalize_unicode(text)
    text = remove_special_chars(text, keep_punctuation=True)
    text = normalize_whitespace(text)
    return text.lower()


def minimal_clean(text: str) -> str:
    """Minimal cleaning for transformer models (preserve casing and structure)."""
    text = clean_html(text)
    text = clean_urls(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


# ==============================================================
# 5. HASHING / DEDUPLICATION
# ==============================================================

def text_hash(text: str) -> str:
    """MD5 hash of normalized text for exact dedup."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    """Count words in text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


# ==============================================================
# 6. DATE UTILITIES
# ==============================================================

def parse_date(date_str: str) -> Optional[date]:
    """Try multiple date formats and return date object or None."""
    if not date_str or not isinstance(date_str, str):
        return None
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y",
        "%Y%m%d", "%B %d, %Y", "%b %d, %Y",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip()[:30], fmt).date()
        except (ValueError, TypeError):
            continue
    # Try pandas as last resort
    try:
        return pd.to_datetime(date_str).date()
    except Exception:
        return None


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return business days between start and end."""
    return pd.bdate_range(start=start, end=end)


def is_valid_date(dt: object, start: str = "2018-01-01", end: str = "2025-12-31") -> bool:
    """Check if a date falls within the project range."""
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt).date()
        elif isinstance(dt, datetime):
            dt = dt.date()
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        return start_dt <= dt <= end_dt
    except Exception:
        return False


# ==============================================================
# 7. NORMALIZATION
# ==============================================================

def minmax_normalize(series: pd.Series, new_min: float = 0, new_max: float = 100) -> pd.Series:
    """Min-max normalize a pandas Series to [new_min, new_max]."""
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series([50.0] * len(series), index=series.index)
    return new_min + (series - s_min) / (s_max - s_min) * (new_max - new_min)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize a pandas Series."""
    return (series - series.mean()) / (series.std() + 1e-8)


# ==============================================================
# 8. FNEI CATEGORIZATION
# ==============================================================

FNEI_CATEGORIES = {
    "Extreme Fear": (0, 20),
    "Fear": (20, 40),
    "Neutral": (40, 60),
    "Greed": (60, 80),
    "Extreme Greed": (80, 100),
}

FNEI_COLORS = {
    "Extreme Fear": "#B71C1C",
    "Fear": "#E53935",
    "Neutral": "#757575",
    "Greed": "#43A047",
    "Extreme Greed": "#1B5E20",
}

EMOTION_COLORS = {
    "fear": "#E53935",
    "greed": "#43A047",
    "uncertainty": "#FB8C00",
    "optimism": "#1E88E5",
}


def fnei_category(score: float) -> str:
    """Map a 0-100 FNEI score to its category label."""
    for cat, (lo, hi) in FNEI_CATEGORIES.items():
        if lo <= score < hi:
            return cat
    return "Extreme Greed" if score >= 80 else "Extreme Fear"


def fnei_color(score: float) -> str:
    """Map a 0-100 FNEI score to its hex color."""
    return FNEI_COLORS.get(fnei_category(score), "#757575")


def get_dominant_emotion(fear: float, greed: float, uncertainty: float, optimism: float) -> str:
    """Return the dominant emotion label given four scores."""
    scores = {"fear": fear, "greed": greed, "uncertainty": uncertainty, "optimism": optimism}
    return max(scores, key=scores.get)


# ==============================================================
# 9. FILE I/O
# ==============================================================

def save_dataframe(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    """Save DataFrame to CSV with logging."""
    path = Path(path)
    ensure_dirs(path.parent)
    df.to_csv(path, index=index)
    logger.info(f"Saved {len(df):,} rows to {path}")


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV to DataFrame with logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def save_json(obj: object, path: Union[str, Path]) -> None:
    """Save dict/list to JSON."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info(f"Saved JSON to {path}")


def load_json(path: Union[str, Path]) -> object:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ==============================================================
# 10. ARTICLE ID GENERATOR
# ==============================================================

def generate_article_id(source: str, date_str: str, headline: str) -> str:
    """Generate deterministic article ID."""
    raw = f"{source}|{date_str}|{headline[:50]}"
    return "ART_" + hashlib.sha1(raw.encode()).hexdigest()[:12].upper()


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded:", cfg["project"]["name"])
    print("Project root:", get_project_root())
