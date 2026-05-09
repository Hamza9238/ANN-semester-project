"""
sentiment_analyzer.py
=====================
Performs sentiment analysis on all collected text (news, Reddit, tweets).

Primary engine  : VADER (vaderSentiment) – lightweight, finance-tuned lexicon.
Labelling scheme:
    score > +0.05  → "positive" (+1)
    score < -0.05  → "negative" (-1)
    otherwise      → "neutral"  (0)

The module also provides a time-windowed aggregation function that computes
mean sentiment per ticker per configurable time window (hourly / 15-minute),
ready to be merged with price features in dataset_builder.py.
"""

import datetime
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TICKER MENTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# Map common company name variants to canonical ticker
TICKER_ALIASES: dict[str, str] = {
    # AAPL
    "apple": "AAPL", "iphone": "AAPL", "ios": "AAPL", "mac": "AAPL",
    # TSLA
    "tesla": "TSLA", "elon": "TSLA", "musk": "TSLA", "ev": "TSLA",
    # GOOGL
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL", "android": "GOOGL",
    # MSFT
    "microsoft": "MSFT", "azure": "MSFT", "windows": "MSFT", "satya": "MSFT",
    # AMZN
    "amazon": "AMZN", "aws": "AMZN", "bezos": "AMZN", "prime": "AMZN",
}


def _preprocess_text(text: str) -> str:
    """
    Clean and normalize text for sentiment analysis.
    
    Operations:
      • Convert to lowercase for consistent analysis
      • Remove URLs (http/https/www patterns)
      • Remove special characters and punctuation (keep spaces)
      • Collapse multiple spaces into single space
      • Strip leading/trailing whitespace
    
    Parameters
    ----------
    text : raw text string
    
    Returns
    -------
    cleaned text ready for VADER analysis
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http, https, www)
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation but keep spaces (VADER needs some structure)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def _detect_tickers(text: str) -> list[str]:
    """
    Return a list of tickers mentioned in *text*.
    Checks for:
      • Direct cashtag  : $AAPL
      • Uppercase match : AAPL (whole-word)
      • Alias match     : "tesla", "apple" (case-insensitive)
    
    Note: Operates on original text (before preprocessing) to catch cashtags and patterns.
    """
    found = set()
    text_lower = text.lower()

    # Cashtag + uppercase
    for ticker in TICKERS:
        pattern = rf"(?<![A-Z]){ticker}(?![A-Z])"
        if re.search(pattern, text) or f"${ticker}" in text:
            found.add(ticker)

    # Aliases
    for alias, ticker in TICKER_ALIASES.items():
        if alias in text_lower:
            found.add(ticker)

    return list(found) if found else ["UNKNOWN"]


# ─────────────────────────────────────────────────────────────────────────────
# VADER SENTIMENT ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Wraps VADER and exposes a clean API for the pipeline.

    Methods
    -------
    analyze(text_items)        → list[dict]  – raw per-document scores
    aggregate(analyzed_items)  → DataFrame   – per-ticker per-time-window features
    """

    # VADER thresholds for positive / negative classification
    POS_THRESHOLD =  0.05
    NEG_THRESHOLD = -0.05

    def __init__(self, time_window: str = "1H"):
        """
        Parameters
        ----------
        time_window : pandas offset alias for aggregation window.
                      "1H" = 1 hour, "15min" = 15 minutes, "1D" = daily.
        """
        self.sia         = SentimentIntensityAnalyzer()
        self.time_window = time_window
        logger.info("SentimentAnalyzer initialised with VADER (time_window=%s)", time_window)

    # ── Per-document analysis ────────────────────────────────────────────────

    def _score_one(self, text: str) -> dict:
        """
        Run VADER on a single text string and return compound + components.
        
        Preprocessing pipeline:
          1. Preprocess text (lowercase, remove URLs, punctuation, extra spaces)
          2. Run VADER sentiment analysis
          3. Classify as positive/negative/neutral based on threshold
        """
        # Preprocess text before VADER analysis
        cleaned_text = _preprocess_text(text)
        
        # Skip if text becomes empty after preprocessing
        if not cleaned_text:
            return {
                "compound":   0.0,
                "pos":        0.0,
                "neu":        0.0,
                "neg":        0.0,
                "label":      "neutral",
                "label_num":  0,
            }
        
        scores = self.sia.polarity_scores(cleaned_text)
        compound = scores["compound"]  # in [-1, 1]

        if compound >= self.POS_THRESHOLD:
            label      = "positive"
            label_num  =  1
        elif compound <= self.NEG_THRESHOLD:
            label      = "negative"
            label_num  = -1
        else:
            label      = "neutral"
            label_num  =  0

        return {
            "compound":   compound,
            "pos":        scores["pos"],
            "neu":        scores["neu"],
            "neg":        scores["neg"],
            "label":      label,
            "label_num":  label_num,
        }

    def analyze(self, text_items: list[dict]) -> list[dict]:
        """
        Analyse every text item collected by data_collector.

        Parameters
        ----------
        text_items : list of {"text", "published", "source"} dicts

        Returns
        -------
        List of enriched dicts – original fields + sentiment scores + tickers.
        """
        logger.info("Running VADER sentiment analysis on %d documents …", len(text_items))
        results = []

        for item in text_items:
            text = item.get("text", "")
            if not text.strip():
                continue

            scores  = self._score_one(text)
            tickers = _detect_tickers(text)

            record = {
                **item,
                **scores,
                "tickers": tickers,
            }
            results.append(record)

        # Summary stats
        pos_count = sum(1 for r in results if r["label"] == "positive")
        neg_count = sum(1 for r in results if r["label"] == "negative")
        neu_count = sum(1 for r in results if r["label"] == "neutral")
        logger.info(
            "  Sentiment distribution -> POS: %d | NEG: %d | NEU: %d",
            pos_count, neg_count, neu_count,
        )
        return results

    # ── Time-windowed aggregation ────────────────────────────────────────────

    def aggregate(
        self,
        analyzed_items: list[dict],
        tickers: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores per ticker per time window.

        For each (ticker, time_window) pair we compute:
          - mean_compound   : average compound score
          - sentiment_sum   : sum of label_num values (+1/-1/0)
          - pos_ratio       : fraction of positive documents
          - neg_ratio       : fraction of negative documents
          - doc_count       : number of documents in the window

        Parameters
        ----------
        analyzed_items : output of self.analyze()
        tickers        : which tickers to aggregate (default: TICKERS)

        Returns
        -------
        DataFrame with MultiIndex or flat columns, indexed by (ticker, time_window).
        Also saved as a flat CSV-friendly DataFrame for merging with price data.
        """
        if tickers is None:
            tickers = TICKERS

        # Explode so each (document, ticker) pair is its own row
        rows = []
        for item in analyzed_items:
            pub = item.get("published")
            if pub is None:
                pub = datetime.datetime.now()
            for ticker in item.get("tickers", []):
                if ticker in tickers:
                    rows.append({
                        "ticker":    ticker,
                        "published": pd.Timestamp(pub),
                        "compound":  item["compound"],
                        "label_num": item["label_num"],
                        "is_pos":    int(item["label"] == "positive"),
                        "is_neg":    int(item["label"] == "negative"),
                    })

        if not rows:
            logger.warning("No ticker-matched sentiment rows – returning empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.set_index("published", inplace=True)
        df.sort_index(inplace=True)

        # Group by ticker, then resample over time window
        all_agg = []
        for ticker in tickers:
            sub = df[df["ticker"] == ticker]
            if sub.empty:
                continue
            resampled = sub.resample(self.time_window).agg(
                mean_compound  = ("compound",  "mean"),
                sentiment_sum  = ("label_num", "sum"),
                pos_ratio      = ("is_pos",    "mean"),
                neg_ratio      = ("is_neg",    "mean"),
                doc_count      = ("compound",  "count"),
            )
            resampled.dropna(how="all", inplace=True)
            resampled["ticker"] = ticker
            all_agg.append(resampled)

        if not all_agg:
            logger.warning("Aggregation produced no rows")
            return pd.DataFrame()

        result = pd.concat(all_agg).reset_index()
        result.rename(columns={"published": "time_window"}, inplace=True)
        result.fillna(0, inplace=True)

        logger.info(
            "Sentiment aggregated: %d rows across %d tickers (window=%s)",
            len(result), result["ticker"].nunique(), self.time_window,
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_sentiment_pipeline(
    text_items: list[dict],
    time_window: str = "1D",
) -> tuple[list[dict], pd.DataFrame]:
    """
    Run the full sentiment pipeline in one call.

    Parameters
    ----------
    text_items  : raw text from data_collector.collect_all_data()
    time_window : aggregation window (default "1D" = daily)

    Returns
    -------
    analyzed_items : enriched per-document list
    sentiment_df   : aggregated sentiment DataFrame
    """
    logger.info("=" * 60)
    logger.info("STEP 2 – SENTIMENT ANALYSIS")
    logger.info("=" * 60)

    analyzer       = SentimentAnalyzer(time_window=time_window)
    analyzed_items = analyzer.analyze(text_items)
    sentiment_df   = analyzer.aggregate(analyzed_items)

    return analyzed_items, sentiment_df


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        {"text": "AAPL absolutely crushed earnings, stock is soaring! 🚀",
         "published": datetime.datetime.now(), "source": "test"},
        {"text": "Tesla faces massive recall – shares tanking pre-market",
         "published": datetime.datetime.now(), "source": "test"},
        {"text": "MSFT trading sideways, no clear direction today",
         "published": datetime.datetime.now(), "source": "test"},
    ]
    analyzed, agg_df = run_sentiment_pipeline(sample_texts, time_window="1D")
    for a in analyzed:
        print(f"[{a['label']:8s}] {a['compound']:+.3f} | {a['text'][:70]}")
    print("\nAggregated sentiment:")
    print(agg_df.to_string(index=False))
