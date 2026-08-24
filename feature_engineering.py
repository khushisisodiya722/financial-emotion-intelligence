"""
feature_engineering.py — NLP Feature Engineering & Emotion Lexicon
============================================================
Implements:
  1. Financial Emotion Lexicon (Fear, Greed, Uncertainty, Optimism)
     - Based on Loughran-McDonald financial word list + custom extensions
  2. Lexicon-based emotion scoring (Method 1)
  3. TF-IDF feature extraction
  4. VADER auxiliary sentiment
  5. Silver label generation for model training
  6. FNEI calculation

References:
  - Loughran, T. & McDonald, B. (2011). When is a liability not a liability?
    Journal of Finance, 66(1), 35-65.
  - Tetlock, P.C. (2007). Giving Content to Investor Sentiment.
    Journal of Finance, 62(3), 1139-1168.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_config, setup_logger, ensure_dirs,
    save_dataframe, load_dataframe, save_json,
    get_dominant_emotion, minmax_normalize, fnei_category
)

logger = setup_logger("feature_engineering")
cfg = load_config()

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / cfg["paths"]["data_processed"]
MODELS_DIR = ROOT / cfg["paths"]["models"]
ensure_dirs(PROC_DIR, MODELS_DIR)


# ==============================================================
# 1. FINANCIAL EMOTION LEXICON
# ==============================================================
# Based on Loughran & McDonald (2011) Financial Sentiment Dictionary
# Extended with India-specific and market-specific terminology
# Reference: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

FINANCIAL_EMOTION_LEXICON = {
    "fear": {
        "crash", "panic", "collapse", "recession", "crisis", "threat", "losses",
        "turmoil", "selloff", "plunge", "slump", "nosedive", "alarming",
        "devastating", "catastrophic", "meltdown", "freefall", "rout", "bloodbath",
        "carnage", "dire", "dread", "bankruptcy", "default", "insolvency",
        "contagion", "systemic", "shock", "trauma", "havoc", "bearish", "downturn",
        "contraction", "stagflation", "hyperinflation", "bust", "decline", "fall",
        "drop", "tumble", "worsen", "deteriorate", "selldown", "bearmarket",
        "depression", "liquidation", "distress", "implosion", "unwind", "capitulate",
        "flee", "flight", "aversion", "warning", "downgrade", "writeoff",
        "writedown", "impairment", "negative", "weak", "vulnerable", "exposure",
        "feared", "fearing", "fearful", "fears", "fear",
        "npa", "stressed", "scam", "fraud", "probe", "raid", "arrested", "defaulter",
        "plummeted", "tumbled", "crashed", "tanked", "sank", "fell", "dropped",
        "struggling", "failing", "failed", "collapsing", "imploding",
    },
    "greed": {
        "rally", "boom", "surge", "record", "bullish", "opportunity", "gains",
        "upside", "momentum", "soar", "climb", "breakout", "milestone", "jackpot",
        "bonanza", "profit", "reward", "outperform", "beat", "exceed",
        "unprecedented", "stellar", "euphoria", "frenzy", "mania",
        "buy", "accumulate", "overweight", "upgrade", "target", "conviction",
        "rush", "chase", "pile", "pour", "flood", "inflow", "appetite",
        "exuberance", "bull", "bullrun", "multibagger", "uptrend",
        "alltime", "peak", "zenith", "historic", "blockbuster", "spectacular",
        "booming", "surging", "rallying", "soaring", "climbing", "skyrocket",
        "fii", "dii", "sip", "ipo", "listing", "gainers", "wealth",
        "returns", "alpha", "outperformed", "soared", "surged", "rallied",
        "jumped", "leaped", "rocketed", "spiked", "gained", "rose",
    },
    "uncertainty": {
        "may", "could", "might", "uncertain", "concerns", "risk", "outlook",
        "possibility", "unclear", "ambiguous", "volatile", "unpredictable",
        "unknown", "caution", "wait", "monitor", "watch", "pending",
        "undecided", "murky", "depends", "conditional", "tentative",
        "speculative", "rumored", "alleged", "headwinds", "challenges",
        "downside", "scenario", "perhaps", "possibly", "potentially",
        "likely", "unlikely", "probable", "improbable", "questionable",
        "doubt", "doubtful", "dubious", "ambivalent", "unsettled",
        "unstable", "erratic", "fluctuating", "oscillating", "mixed",
        "divergent", "split", "divided", "debate", "controversy", "disputed",
        "volatility", "vix", "hedge", "hedging", "cautious", "cautiously",
        "monsoon", "election", "regulatory", "approval", "delayed",
        "uncertainty", "uncertainties", "risks", "unclear", "could",
    },
    "optimism": {
        "growth", "recovery", "strong", "improvement", "expansion", "positive",
        "confidence", "robust", "resilient", "rebound", "upswing", "momentum",
        "promise", "potential", "prospects", "guidance", "stable", "healthy",
        "solid", "sustainable", "consistent", "steady", "constructive",
        "encouraging", "favorable", "supportive", "boost", "optimistic",
        "optimism", "upbeat", "buoyant", "resilience", "turnaround",
        "recover", "revival", "innovation", "disrupt", "transform", "evolve",
        "modernize", "digital", "efficient", "productive", "competitive",
        "export", "surplus", "reform", "develop", "infrastructure", "capex",
        "consumption", "demand", "employment", "jobs", "creation", "income",
        "wealth", "atmanirbhar", "startup", "unicorn", "emerging", "growing",
        "largest", "economy", "gdp", "moderate", "inflows", "investment",
        "stronger", "improving", "rebounding", "recovering", "growing",
        "confident", "bullish", "positive", "upbeat", "flourishing",
    },
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "without", "lack", "lacking", "failed",
    "unable", "cannot", "cant", "wont", "shouldnt", "wouldnt",
    "doesnt", "didnt", "dont", "isnt", "arent", "wasnt", "werent",
}

INTENSIFIERS = {
    "very", "extremely", "significantly", "sharply", "dramatically", "severely",
    "massively", "hugely", "largely", "substantially", "considerably",
    "deeply", "heavily", "seriously", "critically", "violently", "wildly",
}


# ==============================================================
# 2. LEXICON-BASED EMOTION SCORER (METHOD 1)
# ==============================================================

class LexiconEmotionScorer:
    """
    Transparent, rule-based emotion scorer using financial domain lexicon.

    Methodology:
    - Word-level matching with finance domain lexicon
    - Negation window detection (5-word window)
    - Intensifier detection (+50% weight boost)
    - Sentence-level aggregation
    - Article-level aggregation (headline weighted 3x)

    This constitutes Method 1 in the three-method comparison.
    """

    def __init__(self, lexicon: Optional[Dict] = None):
        self.lexicon = lexicon or FINANCIAL_EMOTION_LEXICON
        self.emotions = list(self.lexicon.keys())
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def _get_opposite(self, emotion: str) -> Optional[str]:
        opposites = {
            "fear": "optimism", "greed": "uncertainty",
            "uncertainty": "greed", "optimism": "fear"
        }
        return opposites.get(emotion)

    def _score_tokens(self, tokens: List[str]) -> Dict[str, float]:
        """Score token list with negation and intensifier awareness."""
        scores = {e: 0.0 for e in self.emotions}
        for i, token in enumerate(tokens):
            neg_window = tokens[max(0, i-5):i]
            int_window = tokens[max(0, i-3):i]
            is_negated = any(neg in neg_window for neg in NEGATION_WORDS)
            is_intensified = any(intens in int_window for intens in INTENSIFIERS)
            weight = 1.5 if is_intensified else 1.0

            for emotion, word_set in self.lexicon.items():
                if token in word_set:
                    if is_negated:
                        opposite = self._get_opposite(emotion)
                        if opposite:
                            scores[opposite] += weight * 0.5
                    else:
                        scores[emotion] += weight
        return scores

    def score_text(self, text: str) -> Dict[str, float]:
        """Score a single text string. Returns normalized 0-1 emotion scores."""
        if not isinstance(text, str) or not text.strip():
            return {e: 0.0 for e in self.emotions}

        tokens = text.lower().split()
        raw_scores = self._score_tokens(tokens)
        total_tokens = max(len(tokens), 1)

        # Normalize by length then scale (1% density = 0.5 emotion score)
        normalized = {e: min(1.0, (raw_scores[e] / total_tokens) * 50) for e in self.emotions}

        # Blend with VADER
        if self.vader:
            try:
                vs = self.vader.polarity_scores(text[:500])
                compound = vs["compound"]
                if compound > 0.05:
                    normalized["optimism"] = min(1.0, normalized["optimism"] + compound * 0.3)
                    normalized["greed"] = min(1.0, normalized["greed"] + compound * 0.2)
                elif compound < -0.05:
                    normalized["fear"] = min(1.0, normalized["fear"] + abs(compound) * 0.3)
                    normalized["uncertainty"] = min(1.0, normalized["uncertainty"] + abs(compound) * 0.2)
            except Exception:
                pass

        return normalized

    def score_article(self, row: pd.Series) -> Dict[str, float]:
        """Score article row. Headline is weighted 3x body text."""
        headline_scores = self.score_text(str(row.get("headline", "")))
        body_scores = self.score_text(str(row.get("text_clean", row.get("article_text", ""))))

        combined = {e: (3 * headline_scores[e] + body_scores[e]) / 4 for e in self.emotions}
        combined["dominant_emotion"] = get_dominant_emotion(
            combined["fear"], combined["greed"],
            combined["uncertainty"], combined["optimism"]
        )
        return combined

    def get_emotion_keywords(self, text: str) -> Dict[str, List[str]]:
        """Return which specific words triggered each emotion."""
        if not isinstance(text, str):
            return {e: [] for e in self.emotions}
        tokens = text.lower().split()
        found = {e: [] for e in self.emotions}
        for token in tokens:
            for emotion, word_set in self.lexicon.items():
                if token in word_set and token not in found[emotion]:
                    found[emotion].append(token)
        return found


def score_all_articles(df: pd.DataFrame, scorer: Optional[LexiconEmotionScorer] = None) -> pd.DataFrame:
    """Apply lexicon scoring to all articles."""
    if scorer is None:
        scorer = LexiconEmotionScorer()

    logger.info("Applying lexicon-based emotion scoring...")
    results = df.progress_apply(scorer.score_article, axis=1)
    results_df = pd.DataFrame(list(results))

    df["fear_score"] = results_df["fear"].values
    df["greed_score"] = results_df["greed"].values
    df["uncertainty_score"] = results_df["uncertainty"].values
    df["optimism_score"] = results_df["optimism"].values
    df["dominant_emotion"] = results_df["dominant_emotion"].values
    df["silver_label"] = df["dominant_emotion"]

    logger.info(f"Scored {len(df):,} articles")
    print(df["dominant_emotion"].value_counts().to_string())
    return df


# ==============================================================
# 3. TF-IDF FEATURE EXTRACTION
# ==============================================================

def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: Optional[pd.Series] = None,
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (1, 3),
    save_path: Optional[str] = None
) -> Tuple:
    """
    Build TF-IDF feature matrices.
    Returns (X_train, X_test, vectorizer).
    """
    logger.info(f"Building TF-IDF: max_features={max_features}, ngrams={ngram_range}")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"[a-zA-Z]{2,}",
    )
    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    logger.info(f"TF-IDF train: {X_train.shape}")

    X_test = None
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts.fillna(""))
        logger.info(f"TF-IDF test: {X_test.shape}")

    if save_path:
        joblib.dump(vectorizer, save_path)
        logger.info(f"Vectorizer saved: {save_path}")

    return X_train, X_test, vectorizer


def get_top_tfidf_terms(vectorizer: TfidfVectorizer, n: int = 30) -> pd.DataFrame:
    """Get top TF-IDF terms by IDF score (lower = more common/important)."""
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_
    return pd.DataFrame({"term": feature_names, "idf_score": idf_scores}).sort_values("idf_score").head(n)


# ==============================================================
# 4. FNEI CALCULATION
# ==============================================================

def calculate_fnei(df: pd.DataFrame, formula: str = "weighted",
                   weights: Optional[Dict] = None) -> pd.Series:
    """
    Calculate Financial News Emotion Index (FNEI) — 0 to 100 scale.

    DISCLAIMER: FNEI is a research-created academic index.
    It is NOT an official market indicator and should not be used for
    actual investment decisions.

    Formula candidates tested:
      simple:   (optimism + greed) - (fear + uncertainty)
      weighted: w1*optimism + w2*greed - w3*fear - w4*uncertainty
      ratio:    (optimism + greed) / (fear + uncertainty + optimism + greed)

    Default weights selected based on correlation analysis with India VIX.
    """
    fear = df["fear_score"].fillna(0)
    greed = df["greed_score"].fillna(0)
    uncertainty = df["uncertainty_score"].fillna(0)
    optimism = df["optimism_score"].fillna(0)

    if formula == "simple":
        raw = (optimism + greed) - (fear + uncertainty)
    elif formula == "weighted":
        if weights is None:
            weights = {"optimism": 0.35, "greed": 0.30, "fear": -0.20, "uncertainty": -0.15}
        raw = (
            weights.get("optimism", 0.35) * optimism +
            weights.get("greed", 0.30) * greed +
            weights.get("fear", -0.20) * fear +
            weights.get("uncertainty", -0.15) * uncertainty
        )
    elif formula == "ratio":
        total = (fear + greed + uncertainty + optimism).replace(0, 1)
        raw = (optimism + greed) / total
    else:
        raise ValueError(f"Unknown formula: {formula}")

    return minmax_normalize(raw, 0, 100)


def calculate_daily_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level scores to daily level."""
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date").agg(
        article_count=("article_id", "count"),
        daily_fear=("fear_score", "mean"),
        daily_greed=("greed_score", "mean"),
        daily_uncertainty=("uncertainty_score", "mean"),
        daily_optimism=("optimism_score", "mean"),
        fear_std=("fear_score", "std"),
        greed_std=("greed_score", "std"),
        uncertainty_std=("uncertainty_score", "std"),
        optimism_std=("optimism_score", "std"),
    ).reset_index()

    # Test multiple FNEI formulas
    daily_temp = daily.rename(columns={
        "daily_fear": "fear_score", "daily_greed": "greed_score",
        "daily_uncertainty": "uncertainty_score", "daily_optimism": "optimism_score",
    })
    for formula in ["simple", "weighted", "ratio"]:
        try:
            daily[f"fnei_{formula}"] = calculate_fnei(daily_temp, formula=formula)
        except Exception as e:
            logger.warning(f"FNEI {formula} failed: {e}")

    # Primary FNEI = 7-day rolling weighted FNEI
    if "fnei_weighted" in daily.columns:
        daily["fnei_raw"] = daily["fnei_weighted"]
        daily["fnei"] = daily["fnei_weighted"].rolling(7, min_periods=1).mean()
    elif "fnei_simple" in daily.columns:
        daily["fnei_raw"] = daily["fnei_simple"]
        daily["fnei"] = daily["fnei_simple"].rolling(7, min_periods=1).mean()

    daily["fnei_category"] = daily["fnei"].apply(fnei_category)
    daily = daily.sort_values("date").reset_index(drop=True)

    logger.info(f"Daily emotion scores: {len(daily):,} days")
    return daily


# ==============================================================
# 5. MAIN PIPELINE
# ==============================================================

def run_feature_engineering(
    input_path: Optional[str] = None,
    output_article_path: Optional[str] = None,
    output_daily_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline."""
    if input_path is None:
        input_path = PROC_DIR / "articles_processed.csv"
    if output_article_path is None:
        output_article_path = PROC_DIR / "articles_with_scores.csv"
    if output_daily_path is None:
        output_daily_path = PROC_DIR / "daily_emotions.csv"

    df = load_dataframe(input_path)

    # Score
    scorer = LexiconEmotionScorer()
    df = score_all_articles(df, scorer)

    # Save lexicon
    save_json(
        {k: list(v) for k, v in FINANCIAL_EMOTION_LEXICON.items()},
        PROC_DIR / "emotion_lexicon.json"
    )

    # FNEI per article
    df["fnei"] = calculate_fnei(df, formula="weighted")
    df["fnei_category"] = df["fnei"].apply(fnei_category)

    save_dataframe(df, output_article_path)

    # Daily aggregation
    daily_df = calculate_daily_emotions(df)
    save_dataframe(daily_df, output_daily_path)

    logger.info("Feature engineering complete.")
    return df, daily_df


if __name__ == "__main__":
    df, daily = run_feature_engineering()
    print(df[["date", "headline", "fear_score", "greed_score",
              "uncertainty_score", "optimism_score", "dominant_emotion", "fnei"]].head(10))
    print("\nDaily emotions sample:")
    print(daily.head())
