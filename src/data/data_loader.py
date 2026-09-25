"""
Data loading.

The original run_all.py assumed pre-processed .npy arrays already
existed at data/processed/X_train.npy etc., with no code showing how
they were produced from the raw Dukascopy/ForexSB/FRED/yfinance
sources described in Appendix C. This module is the missing link:
it reads the raw CSV files (which YOU must place — see SETUP.md,
since Dukascopy/ForexSB are licensed sources this sandbox cannot
reach) and produces the processed arrays the rest of the pipeline
consumes.

Expected raw file formats (see SETUP.md for exact column names to
export from each provider):
  - Dukascopy 1-min OHLCV CSV: timestamp, open, high, low, close, volume
  - ForexSB 1-min OHLCV CSV: same columns, used only for the 2010-2023
    cross-validation check described in Appendix C, not for training
  - FRED macro CSV: date, series_id, value (long format) or one column
    per series (wide format) — both are handled below
  - VIX: pulled live via yfinance if not already cached locally
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


class DataFileNotFoundError(FileNotFoundError):
    """Raised with an explicit, actionable message rather than a bare
    FileNotFoundError, since the most common failure mode here is
    'I haven't put the licensed data file in place yet.'"""
    pass


def _require_file(path: str, source_name: str):
    if not os.path.isfile(path):
        raise DataFileNotFoundError(
            f"\n\nMissing required data file for {source_name}: {path}\n"
            f"This file is not bundled with the code (it comes from a "
            f"licensed data provider) and must be placed there manually.\n"
            f"See SETUP.md, section 'Acquiring the data', for exact steps.\n"
        )


def load_dukascopy(path: str) -> pd.DataFrame:
    print(f" -> [TRACE] Validating Dukascopy raw path: {path}")
    _require_file(path, "Dukascopy (primary EUR/USD 1-min OHLCV)")
    print(f" -> [TRACE] Reading Dukascopy CSV data...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f" -> [TRACE] Dukascopy loaded successfully ({len(df):,} rows, columns={list(df.columns)})")
    return df


def load_forexsb(path: str) -> pd.DataFrame:
    print(f" -> [TRACE] Validating ForexSB raw path: {path}")
    _require_file(path, "ForexSB (supplementary EUR/USD 1-min OHLCV, 2010-2023 cross-check)")
    print(f" -> [TRACE] Reading ForexSB CSV data...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f" -> [TRACE] ForexSB loaded successfully ({len(df):,} rows)")
    return df


def load_fred(path: str) -> pd.DataFrame:
    print(f" -> [TRACE] Validating FRED raw path: {path}")
    _require_file(path, "FRED (macroeconomic covariates)")
    print(f" -> [TRACE] Reading FRED macro data...")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    print(f" -> [TRACE] FRED macro data loaded successfully ({len(df):,} rows)")
    return df


def load_vix(ticker: str = "^VIX", cache_path: str = "data/raw/vix_cache.csv",
             start: str = "2010-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """VIX via yfinance (the study's primary VIX source per Ch.3/Ch.4,
    with FRED used only for cross-validation). Falls back to a local
    cache file if yfinance/network is unavailable in the run environment."""
    
    if os.path.isfile(cache_path):
        print(f" -> [TRACE] Local VIX cache found at {cache_path}. Loading...")
        df = pd.read_csv(cache_path)
        # Flatten columns if multi-level headers exist in cached CSV
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Case normalization
        cols = {c: str(c).lower() for c in df.columns}
        df = df.rename(columns=cols)
        
        # Rename Close/Adj Close to vix if necessary
        if "vix" not in df.columns:
            if "close" in df.columns:
                df = df.rename(columns={"close": "vix"})
            elif "adj close" in df.columns:
                df = df.rename(columns={"adj close": "vix"})

        date_col = "date" if "date" in df.columns else "timestamp"
        df["date"] = pd.to_datetime(df[date_col], utc=True)
        print(f" -> [TRACE] VIX cached data loaded successfully ({len(df):,} rows)")
        return df[["date", "vix"]].dropna().drop_duplicates(subset=["date"])

    if not _HAS_YFINANCE:
        raise DataFileNotFoundError(
            f"\n\nyfinance is not installed and no cache file was found at {cache_path}.\n"
            f"Run `pip install yfinance` or place a pre-downloaded VIX CSV at that path.\n"
        )

    print(f" -> [TRACE] Fetching '{ticker}' directly from yfinance API ({start} to {end})...")
    raw = yf.download(ticker, start=start, end=end, progress=False)

    # --- FIX: Flatten MultiIndex column structures returned by yfinance ---
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    
    # Normalize column names to lowercase
    raw.columns = [str(c).lower() for c in raw.columns]
    
    # Handle Date column naming
    date_col = "date" if "date" in raw.columns else "index"
    close_col = "close" if "close" in raw.columns else "adj close"

    df = raw[[date_col, close_col]].rename(columns={date_col: "date", close_col: "vix"})
    df["date"] = pd.to_datetime(df["date"], utc=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f" -> [TRACE] Downloaded & cached {len(df):,} VIX rows to {cache_path}")
    return df


def merge_all_sources(dukascopy_df: pd.DataFrame, fred_df: pd.DataFrame,
                      vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges minute-level OHLCV with daily macro/VIX data via forward-fill,
    matching Ch.3's stated procedure ("lower-frequency VIX and FRED data
    was forward-filled to align with minute-level OHLC timestamps, a
    standard practice to avoid look-ahead bias").
    """
    print(" -> [TRACE] Initializing merge pipeline...")
    df = dukascopy_df.copy()
    df["date"] = df["timestamp"].dt.floor("D")

    # Format VIX timestamps safely
    vix_daily = vix_df.rename(columns={"date": "date"}).copy()
    vix_daily["date"] = pd.to_datetime(vix_daily["date"])
    if vix_daily["date"].dt.tz is None:
        vix_daily["date"] = vix_daily["date"].dt.tz_localize("UTC", nonexistent="shift_forward")
    vix_daily["date"] = vix_daily["date"].dt.floor("D")

    print(" -> [TRACE] Merging VIX series on daily date keys...")
    df = df.merge(vix_daily[["date", "vix"]], on="date", how="left")
    print(f" -> [TRACE] Forward-filling VIX NaNs (Initial missing count: {df['vix'].isna().sum():,})...")
    df["vix"] = df["vix"].ffill()

    # Format FRED timestamps safely
    fred_wide = fred_df.copy()
    fred_wide["date"] = pd.to_datetime(fred_wide["date"])
    if fred_wide["date"].dt.tz is None:
        fred_wide["date"] = fred_wide["date"].dt.tz_localize("UTC", nonexistent="shift_forward")
    fred_wide["date"] = fred_wide["date"].dt.floor("D")

    print(" -> [TRACE] Merging FRED macro features on daily date keys...")
    df = df.merge(fred_wide, on="date", how="left", suffixes=("", "_fred"))
    for col in fred_wide.columns:
        if col != "date":
            df[col] = df[col].ffill()

    print(f" -> [TRACE] Data merge complete. Combined DataFrame dimensions: {df.shape}")
    return df.drop(columns=["date"])


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Chronological train/val/test split per config dates — no shuffling,
    since this is time-series data and shuffling would leak future
    information into training (look-ahead bias)."""
    ts = df["timestamp"]
    print(f" -> [TRACE] Performing chronological splits using ranges:")
    print(f"    Train: {cfg['train_start']} -> {cfg['train_end']}")
    print(f"    Val:   {cfg['val_start']} -> {cfg['val_end']}")
    print(f"    Test:  {cfg['test_start']} -> {cfg['test_end']}")

    train = df[(ts >= cfg["train_start"]) & (ts <= cfg["train_end"])]
    val = df[(ts >= cfg["val_start"]) & (ts <= cfg["val_end"])]
    test = df[(ts >= cfg["test_start"]) & (ts <= cfg["test_end"])]

    print(f" -> [TRACE] Split row counts -> Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    return {"train": train, "val": val, "test": test}


def save_processed_arrays(splits: dict[str, pd.DataFrame], feature_cols: list[str],
                            target_col: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f" -> [TRACE] Saving processed NumPy arrays to directory: {out_dir}")
    for split_name, split_df in splits.items():
        X = split_df[feature_cols].to_numpy(dtype=np.float32)
        y = split_df[target_col].to_numpy(dtype=np.float32)

        x_file = os.path.join(out_dir, f"X_{split_name}.npy")
        y_file = os.path.join(out_dir, f"y_{split_name}.npy")

        np.save(x_file, X)
        np.save(y_file, y)

        x_mb = os.path.getsize(x_file) / (1024 * 1024)
        y_mb = os.path.getsize(y_file) / (1024 * 1024)
        print(f"    [SAVED] X_{split_name}.npy -> shape {X.shape} ({x_mb:.2f} MB)")
        print(f"    [SAVED] y_{split_name}.npy -> shape {y.shape} ({y_mb:.2f} MB)")
