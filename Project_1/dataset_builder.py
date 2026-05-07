"""
dataset_builder.py
==================
Merges stock price features with aggregated sentiment scores into a single
supervised-learning dataset suitable for sequence models (RNN / LSTM / GRU).

Pipeline
--------
1. For each ticker take the daily price DataFrame (from data_collector).
2. Align sentiment aggregation windows to trading days via forward-fill.
3. Normalize all numeric features with StandardScaler.
4. Build sliding windows of length SEQ_LEN days (input sequences).
5. The target is the binary next-day direction (up=1, down=0).
6. Split into train / validation / test sets chronologically (no leakage).
7. Convert everything to PyTorch tensors for model training.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Number of past time steps each sample looks back
SEQ_LEN = 10

# Train / val / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# Price features used as model inputs
PRICE_FEATURES = [
    "Close", "Volume", "MA5", "MA20", "Daily_Return", "Volatility",
]

# Sentiment features merged from sentiment_analyzer output
SENTIMENT_FEATURES = [
    "mean_compound", "sentiment_sum", "pos_ratio", "neg_ratio", "doc_count",
]


# ─────────────────────────────────────────────────────────────────────────────
# MERGE  –  price + sentiment → unified daily DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _merge_ticker(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    For a single ticker, left-join price data with sentiment data on date,
    then forward-fill missing sentiment windows.

    The price DataFrame has a DatetimeIndex (trading days).
    The sentiment DataFrame has columns [time_window, ticker, ...].
    """
    # Pull this ticker's sentiment rows and index by date
    sent_sub = sentiment_df[sentiment_df["ticker"] == ticker].copy()

    if sent_sub.empty:
        logger.warning("No sentiment data for %s – filling with zeros", ticker)
        for col in SENTIMENT_FEATURES:
            price_df[col] = 0.0
        return price_df

    sent_sub["date"] = pd.to_datetime(sent_sub["time_window"]).dt.normalize()
    sent_sub = sent_sub.drop_duplicates(subset="date").set_index("date")

    # Left join on date
    merged = price_df.copy()
    merged.index = pd.to_datetime(merged.index).normalize()

    for col in SENTIMENT_FEATURES:
        if col in sent_sub.columns:
            merged[col] = merged.index.map(
                lambda d: sent_sub.loc[d, col] if d in sent_sub.index else np.nan
            )
        else:
            merged[col] = 0.0

    # Forward-fill then back-fill remaining NaNs
    merged[SENTIMENT_FEATURES] = (
        merged[SENTIMENT_FEATURES]
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER  –  sliding windows of length SEQ_LEN
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a flat DataFrame into overlapping windows.

    Returns
    -------
    X      : shape (n_samples, seq_len, n_features)  – input sequences
    y_cls  : shape (n_samples,)                       – binary direction target
    y_reg  : shape (n_samples,)                       – return magnitude target
    """
    feature_arr = df[feature_cols].values          # (T, F)
    target_cls  = df["Target"].values               # (T,) int
    target_reg  = df["Return_Magnitude"].values     # (T,) float

    X, y_cls, y_reg = [], [], []
    for i in range(len(feature_arr) - seq_len):
        X.append(feature_arr[i : i + seq_len])
        y_cls.append(target_cls[i + seq_len])
        y_reg.append(target_reg[i + seq_len])

    return (
        np.array(X,     dtype=np.float32),
        np.array(y_cls, dtype=np.int64),
        np.array(y_reg, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data only, then apply to val + test.
    Works on 3-D arrays (n, seq, features) by reshaping.
    """
    n_tr, seq, feat = X_train.shape
    n_va            = X_val.shape[0]
    n_te            = X_test.shape[0]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, feat)
    X_val_2d   = X_val.reshape(-1, feat)
    X_test_2d  = X_test.reshape(-1, feat)

    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(n_tr, seq, feat)
    X_val   = scaler.transform(X_val_2d).reshape(n_va, seq, feat)
    X_test  = scaler.transform(X_test_2d).reshape(n_te, seq, feat)

    return X_train, X_val, X_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Minimal PyTorch Dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BUILDER  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    price_data:    dict,
    sentiment_df:  pd.DataFrame,
    seq_len:       int = SEQ_LEN,
    batch_size:    int = 32,
    tickers:       Optional[list[str]] = None,
    save_data:     bool = True,
) -> tuple[dict, int]:
    """
    Build train / val / test DataLoaders from price + sentiment data.

    Parameters
    ----------
    price_data   : dict[ticker -> DataFrame] from data_collector
    sentiment_df : aggregated sentiment from sentiment_analyzer
    seq_len      : look-back window length
    batch_size   : DataLoader batch size
    tickers      : which tickers to include (default: all)
    save_data    : whether to save the merged dataset to CSV

    Returns
    -------
    loaders : dict with keys "train", "val", "test" and sub-keys per ticker
    n_features : number of input features (for model initialisation)
    """
    logger.info("=" * 60)
    logger.info("STEP 3 – DATASET BUILDING")
    logger.info("=" * 60)

    if tickers is None:
        tickers = TICKERS

    feature_cols = PRICE_FEATURES + SENTIMENT_FEATURES
    all_loaders  = {}

    for ticker in tickers:
        if ticker not in price_data:
            logger.warning("  Skipping %s – no price data", ticker)
            continue

        df = _merge_ticker(price_data[ticker], sentiment_df, ticker)

        # Drop rows with any NaN in feature columns
        df = df[feature_cols + ["Target", "Return_Magnitude"]].dropna()

        if save_data:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, f"{ticker}_processed_dataset.csv")
            df.to_csv(csv_path)
            logger.info("  %s: Saved processed dataset to %s", ticker, csv_path)

        if len(df) < seq_len + 5:
            logger.warning("  %s: not enough rows (%d) – skipping", ticker, len(df))
            continue

        X, y_cls, y_reg = _build_sequences(df, feature_cols, seq_len)

        # Chronological split (no shuffling to avoid look-ahead)
        n        = len(X)
        n_train  = int(n * TRAIN_RATIO)
        n_val    = int(n * VAL_RATIO)

        X_train,  X_val,  X_test  = X[:n_train],  X[n_train:n_train+n_val],  X[n_train+n_val:]
        yc_train, yc_val, yc_test = y_cls[:n_train], y_cls[n_train:n_train+n_val], y_cls[n_train+n_val:]
        yr_train, yr_val, yr_test = y_reg[:n_train], y_reg[n_train:n_train+n_val], y_reg[n_train+n_val:]

        # Guard against empty splits
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            logger.warning("  %s: split produced empty partition – skipping", ticker)
            continue

        X_train, X_val, X_test, _ = _normalise(X_train, X_val, X_test)

        train_ds = TimeSeriesDataset(X_train, yc_train, yr_train)
        val_ds   = TimeSeriesDataset(X_val,   yc_val,   yr_val)
        test_ds  = TimeSeriesDataset(X_test,  yc_test,  yr_test)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=1,          shuffle=False)

        all_loaders[ticker] = {
            "train": train_loader,
            "val":   val_loader,
            "test":  test_loader,
            "n_samples": (len(train_ds), len(val_ds), len(test_ds)),
        }

        logger.info(
            "  %s: %d train | %d val | %d test sequences",
            ticker, len(train_ds), len(val_ds), len(test_ds),
        )

    if not all_loaders:
        raise RuntimeError("No usable ticker datasets were built. Check data quality.")

    # Determine n_features from the first ticker's training batch
    first_ticker = next(iter(all_loaders))
    first_batch  = next(iter(all_loaders[first_ticker]["train"]))
    n_features   = first_batch[0].shape[-1]   # (batch, seq, features)

    logger.info("\nDataset ready. Input shape: (batch, %d, %d)", seq_len, n_features)
    return all_loaders, n_features


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector   import collect_all_data
    from sentiment_analyzer import run_sentiment_pipeline

    price_data, text_data     = collect_all_data()
    _, sentiment_df           = run_sentiment_pipeline(text_data)
    loaders, n_feat           = build_dataset(price_data, sentiment_df)

    for ticker, ld in loaders.items():
        batch = next(iter(ld["train"]))
        print(f"{ticker}: batch X shape={batch[0].shape}, y_cls={batch[1][:4]}")
    print(f"n_features = {n_feat}")
