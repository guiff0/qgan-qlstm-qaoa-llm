"""
Data loading and multi-source merge.

Reads the raw OHLCV / FRED files and the yfinance VIX feed, and produces the
merged frame the rest of the pipeline consumes.

Expected raw formats (see SETUP.md):
  - Dukascopy 1-min OHLCV CSV: timestamp, open, high, low, close, volume
  - ForexSB 1-min OHLCV CSV: same columns (2010-2023 cross-check only)
  - FRED macro CSV: wide (date + one column per series) OR long
    (date, series_id, value) -- both are handled
  - VIX: yfinance, cached locally

INGESTION FIXES IN THIS VERSION (each one silently corrupted or crashed a run):

  merge_all_sources
    1. LOOK-AHEAD BIAS. Daily values were joined on the same calendar day, so
       a 00:00 UTC minute bar received that day's VIX *close* (published ~21:00
       UTC) and a monthly FRED value dated the 1st (CPI/UNRATE are published
       5-6 weeks later). Ch.3 claims forward-filling avoids look-ahead; it only
       does if each observation is applied from when it was actually knowable.
       Values are now joined from `obs_date + publication_lag`, per series.
    2. FRED long format was documented as supported but never handled: the
       string column `series_id` and a column literally named `value` ended up
       in the feature matrix. Long format is now pivoted to wide.
    3. A FRED column colliding with an OHLCV/VIX column (e.g. `vix`) was
       suffixed `_fred` by merge(), but the following loop then forward-filled
       the *unsuffixed* (OHLCV) column instead. Collisions are renamed
       explicitly and never touch the price columns.
    4. Leading rows with no observation available yet are trimmed here (the
       lag in (1) makes them weeks long -- months if the FRED file has no
       lookback before the study start; acquire_all_data.py now pulls one year
       early). prepare_data.py's dropna() would have removed them anyway, so
       this is about doing it explicitly and reporting the count.

  load_vix (yfinance)
    5. `end` is EXCLUSIVE in yfinance, so end="2025-12-31" never returned
       Dec 31. It is now inclusive.
    6. yfinance does not raise on a rate-limit/network failure; it returns an
       empty frame. The old code either died with an opaque KeyError (no
       retry), or -- when the empty frame still carried its column headers --
       returned 0 rows and WROTE a header-only cache that every later run then
       trusted without ever retrying the network (reproduced against the
       original). Downloads are now retried with backoff, validated non-empty
       and covering the window, and only then cached.
    7. The cache was trusted regardless of the requested date range. It is now
       checked for coverage and re-fetched when stale/partial.
    8. Cache writes are atomic (temp file + os.replace).
    9. yfinance's multi-row CSV header ('Price/Ticker/Date' rows) and tz-aware
       dates are handled; junk rows are coerced away and non-positive values
       dropped, instead of crashing or leaking strings into `vix`.

  load_dukascopy / load_forexsb
   10. Duplicate timestamps were never removed (the report says they are).
       Required columns are validated, prices coerced to numeric, unparsable or
       non-positive rows dropped and counted.
"""
from __future__ import annotations

import os
import time
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.io import atomic_write_text

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    yf = None
    _HAS_YFINANCE = False


class DataFileNotFoundError(FileNotFoundError):
    """Raised with an explicit, actionable message rather than a bare
    FileNotFoundError, since the most common failure mode here is
    'I haven't put the licensed data file in place yet.'"""
    pass


# Days after a series' observation date before the value is treated as
# *available* to a forecaster (conservative). FRED monthly series are dated
# the 1st of the month they describe. Matched case-insensitively; anything not
# listed uses `default_lag_days`. Override via merge_all_sources(release_lags=).
# NOTE: FRED serves *revised* values; only ALFRED vintages remove revision
# look-ahead. This handles publication-timing look-ahead, not revisions.
DEFAULT_FRED_RELEASE_LAGS_DAYS: dict[str, int] = {
    "FEDFUNDS": 35,   # H.15 monthly average, published early next month
    "GS10": 35,       # monthly average of 10y yield
    "CPIAUCSL": 45,   # CPI published ~mid next month
    "UNRATE": 38,     # jobs report, first Friday of next month
    "GDP": 120,       # quarterly, dated 1st day of quarter; advance estimate ~4 months later
}
DEFAULT_VIX_LAG_DAYS = 1  # daily close known only after the session ends


def _require_file(path: str, source_name: str):
    if not os.path.isfile(path):
        raise DataFileNotFoundError(
            f"\n\nMissing required data file for {source_name}: {path}\n"
            f"This file is not bundled with the code (it comes from a "
            f"licensed data provider) and must be placed there manually.\n"
            f"See SETUP.md, section 'Acquiring the data', for exact steps.\n"
        )


# ----------------------------------------------------------------------
# OHLCV (Dukascopy / ForexSB)
# ----------------------------------------------------------------------
_TS_ALIASES = ("timestamp", "datetime", "date", "time", "unnamed: 0")


def _load_ohlcv(path: str, source_name: str) -> pd.DataFrame:
    _require_file(path, source_name)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" not in df.columns:
        alias = next((a for a in _TS_ALIASES if a in df.columns), None)
        if alias is None:
            raise ValueError(f"{source_name}: no timestamp column in {path}; found {list(df.columns)}")
        df = df.rename(columns={alias: "timestamp"})

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns {missing}; found {list(df.columns)}")
    if "volume" not in df.columns:
        print(f" -> [WARN] {source_name}: no 'volume' column in {path}")

    n_raw = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    num_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df["timestamp"].isna() | df[["open", "high", "low", "close"]].isna().any(axis=1)
    bad |= (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        print(f" -> [WARN] {source_name}: dropping {n_bad:,} rows with unparsable timestamp / "
              f"non-numeric or non-positive OHLC")
        if n_bad > 0.01 * max(n_raw, 1):
            raise ValueError(f"{source_name}: {n_bad:,}/{n_raw:,} rows invalid (>1%) -- "
                             f"check the file's column layout/timestamp format before proceeding.")
    df = df.loc[~bad]

    df = df.sort_values("timestamp", kind="stable")
    n_dup = int(df["timestamp"].duplicated(keep="last").sum())
    if n_dup:
        print(f" -> [WARN] {source_name}: dropping {n_dup:,} duplicate timestamps (keeping last)")
        df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.reset_index(drop=True)
    print(f" -> [TRACE] {source_name} loaded ({len(df):,} rows, {df['timestamp'].iloc[0]} -> "
          f"{df['timestamp'].iloc[-1]})")
    return df


def load_dukascopy(path: str) -> pd.DataFrame:
    print(f" -> [TRACE] Reading Dukascopy CSV: {path}")
    return _load_ohlcv(path, "Dukascopy (primary EUR/USD 1-min OHLCV)")


def load_forexsb(path: str) -> pd.DataFrame:
    print(f" -> [TRACE] Reading ForexSB CSV: {path}")
    return _load_ohlcv(path, "ForexSB (supplementary EUR/USD 1-min OHLCV, 2010-2023 cross-check)")


# ----------------------------------------------------------------------
# FRED
# ----------------------------------------------------------------------
def load_fred(path: str) -> pd.DataFrame:
    """Returns a frame with a UTC `date` column plus one numeric column per
    series (long-format files are pivoted; FRED's '.' missing marker -> NaN)."""
    print(f" -> [TRACE] Reading FRED macro data: {path}")
    _require_file(path, "FRED (macroeconomic covariates)")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c: c.lower() for c in df.columns}
    if "date" not in lower.values():
        raise ValueError(f"FRED file {path} has no 'date' column; found {list(df.columns)}")
    df = df.rename(columns={c: "date" for c, l in lower.items() if l == "date"})
    df = df.rename(columns={c: l for c, l in lower.items() if l in ("series_id", "value")})

    if {"series_id", "value"} <= set(df.columns):  # long format
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = (df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
                .reset_index())
        df.columns.name = None
        print(f" -> [TRACE] FRED long format pivoted to wide: series={[c for c in df.columns if c != 'date']}")
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    print(f" -> [TRACE] FRED macro data loaded ({len(df):,} rows)")
    return df


# ----------------------------------------------------------------------
# VIX (yfinance)
# ----------------------------------------------------------------------
_DATE_CANDIDATES = ("date", "datetime", "timestamp", "index", "price")


def _normalize_vix_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Any yfinance-shaped frame or cached CSV -> clean (date[UTC midnight], vix).

    Handles MultiIndex columns, mixed case, 'Close' vs 'Adj Close', tz-aware
    dates, and the junk 'Ticker'/'Date' header rows that yfinance's multi-level
    CSV export leaves in the first data rows (coerced to NaT/NaN and dropped)."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns and df.index.name is not None:
        df = df.reset_index()
        df.columns = [str(c).strip().lower() for c in df.columns]

    if "vix" not in df.columns:
        for cand in ("close", "adj close"):
            if cand in df.columns:
                df = df.rename(columns={cand: "vix"})
                break
    if "vix" not in df.columns:
        raise ValueError(f"No VIX/Close column found; columns={list(df.columns)}")

    date_col = next((c for c in _DATE_CANDIDATES if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No date column found; columns={list(df.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.floor("D"),
        "vix": pd.to_numeric(df["vix"], errors="coerce"),
    })
    n0 = len(out)
    out = out.dropna()
    out = out[out["vix"] > 0]  # VIX is strictly positive; 0 is a bad-tick/placeholder
    if n0 - len(out):
        print(f" -> [TRACE] VIX: dropped {n0 - len(out):,} non-parsable / non-positive rows")
    return (out.sort_values("date").drop_duplicates(subset="date", keep="last")
               .reset_index(drop=True))


def _download_vix(ticker: str, start: str, end: str, max_retries: int) -> pd.DataFrame:
    if not _HAS_YFINANCE:
        raise DataFileNotFoundError(
            "\n\nyfinance is not installed and no usable cache exists.\n"
            "Run `pip install yfinance` or place a pre-downloaded VIX CSV at the cache path.\n"
        )
    # yfinance's `end` is exclusive; add a day so `end` itself is included.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f" -> [TRACE] yfinance '{ticker}' {start} -> {end} (attempt {attempt}/{max_retries})")
            raw = yf.download(ticker, start=start, end=end_excl, progress=False,
                              auto_adjust=False, threads=False)
            if raw is not None and len(raw) > 0:
                return _normalize_vix_frame(raw)
            last_err = RuntimeError("yfinance returned an empty frame "
                                    "(rate limit, network failure, or bad ticker)")
        except Exception as e:  # noqa: BLE001 - yfinance raises many types
            last_err = e
        if attempt < max_retries:
            time.sleep(2 ** attempt)
    raise DataFileNotFoundError(
        f"\n\nCould not download {ticker} from yfinance after {max_retries} attempts: {last_err}\n"
        f"Nothing was written to the cache. Check your network, retry later, or place a "
        f"pre-downloaded VIX CSV at the cache path.\n"
    ) from last_err


def _cache_covers(df: pd.DataFrame, start: str, end: str, tol_days: int = 10) -> tuple[bool, str]:
    """A cache is usable only if it spans the requested window (tolerating
    weekends/holidays at the edges, and clipping `end` to today)."""
    if df.empty:
        return False, "cache is empty"
    want_start = pd.Timestamp(start, tz="UTC")
    want_end = min(pd.Timestamp(end, tz="UTC"), pd.Timestamp.now(tz="UTC").floor("D"))
    have_start, have_end = df["date"].min(), df["date"].max()
    if have_start > want_start + pd.Timedelta(days=tol_days):
        return False, f"cache starts {have_start.date()}, need <= {start}"
    if have_end < want_end - pd.Timedelta(days=tol_days):
        return False, f"cache ends {have_end.date()}, need >= {want_end.date()}"
    return True, "ok"


def load_vix(ticker: str = "^VIX", cache_path: str = "data/raw/vix_cache.csv",
             start: str = "2010-01-01", end: str = "2025-12-31",
             max_retries: int = 3, force_refresh: bool = False) -> pd.DataFrame:
    """VIX daily close via yfinance (primary source per Ch.3/Ch.4), cached.

    Uses the cache only when it covers [start, end]; otherwise re-downloads.
    Never writes an empty/failed download to the cache, and never silently
    falls back to a partial cache."""
    cached: Optional[pd.DataFrame] = None
    if os.path.isfile(cache_path) and not force_refresh:
        try:
            cached = _normalize_vix_frame(pd.read_csv(cache_path))
        except Exception as e:  # corrupt / unrecognized cache -> re-fetch
            print(f" -> [WARN] VIX cache {cache_path} unreadable ({e}); will re-download")
        if cached is not None:
            ok, why = _cache_covers(cached, start, end)
            if ok:
                print(f" -> [TRACE] VIX cache OK ({len(cached):,} rows) from {cache_path}")
                return _clip(cached, start, end)
            print(f" -> [WARN] VIX cache rejected: {why}; re-downloading")

    df = _download_vix(ticker, start, end, max_retries)
    ok, why = _cache_covers(df, start, end)
    if not ok:
        raise DataFileNotFoundError(f"\n\nDownloaded VIX data does not cover the requested window: {why}\n"
                                    f"Not cached.\n")
    atomic_write_text(cache_path, df.to_csv(index=False))
    print(f" -> [TRACE] Downloaded & cached {len(df):,} VIX rows to {cache_path}")
    return _clip(df, start, end)


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    out = df[(df["date"] >= lo) & (df["date"] < hi)].reset_index(drop=True)
    gaps = out["date"].diff().dt.days.dropna()
    if len(gaps) and gaps.max() > 10:
        print(f" -> [WARN] VIX has a {int(gaps.max())}-day gap ending "
              f"{out.loc[gaps.idxmax(), 'date'].date()}")
    return out


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------
def _asof_available(ts: np.ndarray, obs_dates: np.ndarray, values: np.ndarray,
                    lag_days: float) -> tuple[np.ndarray, np.ndarray]:
    """For each minute timestamp `ts`, return the most recent observation whose
    availability time (obs_date + lag) is <= ts, and its age in days.
    NaN / NaT where nothing is available yet. Equivalent to a lagged
    forward-fill, without merge_asof's dtype/unit strictness."""
    avail = obs_dates + np.timedelta64(int(round(lag_days * 86400)), "s")
    order = np.argsort(avail, kind="stable")
    avail, vals, obs = avail[order], values[order], obs_dates[order]
    idx = np.searchsorted(avail, ts, side="right") - 1
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    out = np.where(ok, vals[safe], np.nan)
    age_days = np.where(ok, (ts - obs[safe]) / np.timedelta64(1, "D"), np.nan)
    return out.astype("float64"), age_days


def _utc_naive_ns(s: pd.Series) -> np.ndarray:
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def merge_all_sources(dukascopy_df: pd.DataFrame, fred_df: pd.DataFrame,
                      vix_df: pd.DataFrame, *,
                      release_lags: Optional[Mapping[str, float]] = None,
                      vix_lag_days: float = DEFAULT_VIX_LAG_DAYS,
                      default_lag_days: float = 1.0,
                      max_staleness_days: float = 100.0,
                      drop_leading_incomplete: bool = True) -> pd.DataFrame:
    """
    Aligns daily/monthly VIX and FRED series onto minute bars, WITHOUT
    look-ahead: each observation applies only from (obs_date + publication
    lag) onward, then is held (forward-filled) until superseded.

    release_lags: {series_name: days}, case-insensitive; overrides
        DEFAULT_FRED_RELEASE_LAGS_DAYS. Set every lag to 0 (and vix_lag_days=0)
        to reproduce the legacy same-day join.
    max_staleness_days: warn if a held value is more than this many days older
        than its publication lag (i.e. the series stopped updating; 100 covers a
        quarterly series plus slack).
    drop_leading_incomplete: trim the leading contiguous rows for which some
        merged series has no available observation yet.
    """
    print(" -> [TRACE] Initializing merge pipeline (publication-lag aware)...")
    df = dukascopy_df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    ts = _utc_naive_ns(df["timestamp"])
    n_in = len(df)

    lags = {k.upper(): v for k, v in DEFAULT_FRED_RELEASE_LAGS_DAYS.items()}
    lags.update({k.upper(): v for k, v in (release_lags or {}).items()})

    added: list[str] = []

    def _attach(name: str, obs_dates: np.ndarray, values: np.ndarray, lag: float):
        m = ~np.isnan(values)  # hold the last *valid* observation, as ffill would
        if not m.any():
            raise ValueError(f"Series '{name}' has no valid observations")
        out_name = name if name not in df.columns else f"{name}_fred"
        if out_name in df.columns:
            raise ValueError(f"Column name collision on '{out_name}'")
        vals, age = _asof_available(ts, obs_dates[m], values[m], lag)
        df[out_name] = vals
        added.append(out_name)
        finite = age[np.isfinite(age)]
        max_age = float(finite.max()) if len(finite) else float("nan")
        n_lead = int(np.isnan(vals).sum())
        print(f"    - {out_name}: lag={lag:g}d, leading-unavailable rows={n_lead:,}, max age={max_age:.0f}d")
        if max_age > lag + max_staleness_days:
            print(f" -> [WARN] '{out_name}' holds a value up to {max_age:.0f} days old "
                  f"(lag {lag:g}d + slack {max_staleness_days:g}d); check for a gap in the source series.")

    vix = vix_df[["date", "vix"]].copy()
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.dropna().sort_values("date")
    _attach("vix", _utc_naive_ns(vix["date"]).astype("datetime64[D]").astype("datetime64[ns]"),
            vix["vix"].to_numpy(dtype="float64"), vix_lag_days)

    fred = fred_df.copy()
    fred_dates = _utc_naive_ns(fred["date"]).astype("datetime64[D]").astype("datetime64[ns]")
    for col in [c for c in fred.columns if c != "date"]:
        vals = pd.to_numeric(fred[col], errors="coerce").to_numpy(dtype="float64")
        _attach(col, fred_dates, vals, lags.get(str(col).upper(), default_lag_days))

    if drop_leading_incomplete and added:
        complete = ~df[added].isna().any(axis=1).to_numpy()
        first = int(np.argmax(complete)) if complete.any() else len(df)
        if first:
            print(f" -> [TRACE] Trimming {first:,} leading rows with no available macro/VIX observation yet")
            df = df.iloc[first:].reset_index(drop=True)

    print(f" -> [TRACE] Merge complete: {n_in:,} -> {len(df):,} rows, shape {df.shape}")
    return df


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Chronological train/val/test split per config dates — no shuffling,
    since this is time-series data and shuffling would leak future
    information into training (look-ahead bias).

    BUG FIXED: config dates are calendar days and the *_end dates are meant
    inclusive, but `ts <= "2021-12-31"` compares against 2021-12-31 00:00, so
    every bar from the final day of each period matched NO split (train_end
    2021-12-31 vs val_start 2022-01-01 left a one-day hole, likewise val/test
    and the tail of the test window). End bounds now cover the whole day."""
    ts = df["timestamp"]
    day = pd.Timedelta(days=1)

    def _utc(d):
        t = pd.Timestamp(d)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    print(f" -> [TRACE] Performing chronological splits using ranges (end dates inclusive):")
    print(f"    Train: {cfg['train_start']} -> {cfg['train_end']}")
    print(f"    Val:   {cfg['val_start']} -> {cfg['val_end']}")
    print(f"    Test:  {cfg['test_start']} -> {cfg['test_end']}")

    def _slice(a, b):
        return df[(ts >= _utc(cfg[a])) & (ts < _utc(cfg[b]) + day)]

    train = _slice("train_start", "train_end")
    val = _slice("val_start", "val_end")
    test = _slice("test_start", "test_end")

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
