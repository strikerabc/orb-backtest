"""
data_layer.py — data acquisition, caching, and enrichment.

Public API
----------
ensure_data(sym, api_key=None) -> pd.DataFrame
    Returns merged 1-minute bars for the symbol across the full history.
    Checks local cache first; builds it from existing parquet + Databento
    download if missing. Adds 4h-ATR and per-bar enrichment columns.

ensure_daily(sym, api_key=None) -> pd.DataFrame
    Same but for 1-day bars (used for gap/vol enrichment joined to trades).
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    ATR_ANCHOR_ET, ATR_BAR_MINUTES, ATR_PERIOD,
    DATASET, DOWNLOAD_END, DOWNLOAD_END_BUFFER_DAYS, DOWNLOAD_START,
    EXISTING_DATA_ROOT, SCHEMA_1D, SCHEMA_1M, STYPE,
    INSTRUMENTS, DATA_DIR,
)

log = logging.getLogger("orb.data")


def safe_end_date(buffer_days: int = DOWNLOAD_END_BUFFER_DAYS) -> str:
    """
    Latest end date that stays inside GLBX.MDP3 historical licensing.

    end = today raises 422 dataset_unavailable_range near the real-time
    boundary, so back off DOWNLOAD_END_BUFFER_DAYS. `end` is exclusive.
    """
    ts = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=buffer_days)
    return ts.strftime("%Y-%m-%d")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA = _PROJECT_ROOT / DATA_DIR
_DATA.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def _roll_tag(sym: str) -> str:
    """
    Roll type from the instrument's continuous symbol: 'c', 'v' or 'n'.
    e.g. "GC.v.0" -> "v".  Defaults to 'c' if the symbol is malformed.
    """
    parts = INSTRUMENTS[sym]["continuous_symbol"].split(".")
    return parts[1] if len(parts) >= 2 and parts[1] in ("c", "v", "n") else "c"


def _cache_path(sym: str, schema: str) -> Path:
    """
    Cache path, keyed by ROLL TYPE as well as symbol.

    Why the roll must be in the filename: the original scheme was
    "{sym}_{tag}.parquet", so changing continuous_symbol from GC.c.0 to
    GC.v.0 left ensure_data() finding the old file and silently returning the
    broken .c.0 data. The config change would appear to work and do nothing.

    Legacy fallback: files written before this change came from .c.0 pulls, so
    they remain valid for instruments still on .c.0 and are reused as-is (no
    re-download for ES/NQ/RTY/CL/BTC/ETH). Instruments switched to .v.0/.n.0
    resolve to a new path, which triggers a fresh download and leaves the old
    file untouched on disk.
    """
    tag = "1m" if schema == SCHEMA_1M else "1d"
    roll = _roll_tag(sym)

    versioned = _DATA / f"{sym}_{roll}_{tag}.parquet"
    if versioned.exists():
        return versioned

    legacy = _DATA / f"{sym}_{tag}.parquet"
    if legacy.exists() and roll == "c":
        return legacy

    return versioned


def _load_existing_1m(sym: str) -> pd.DataFrame:
    """Load monthly parquet files from the prior Databento pull (2023-2026)."""
    root = Path(EXISTING_DATA_ROOT) / sym
    if not root.exists():
        log.warning("Existing data root not found: %s", root)
        return pd.DataFrame()

    frames = []
    for ydir in sorted(root.iterdir()):
        if not ydir.is_dir():
            continue
        for f in sorted(ydir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "_contract"
                ])
                frames.append(df)
            except Exception as exc:
                log.warning("Skip %s: %s", f, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # forward-fill missing contract labels (ES has ~5% null in roll months)
    df["_contract"] = df["_contract"].ffill()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    log.info("Existing %s: %d bars %s -> %s", sym, len(df),
             df.timestamp.min(), df.timestamp.max())
    return df.reset_index(drop=True)


def _download_databento(sym: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Download ohlcv-1m bars from Databento GLBX.MDP3, return normalized DataFrame."""
    try:
        import databento as db
    except ImportError:
        raise ImportError("pip install databento")

    log.info("Downloading %s %s → %s from Databento...", sym, start, end)
    client = db.Historical(key=api_key)

    # Print cost estimate before downloading — never spend credits silently.
    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=[INSTRUMENTS[sym]["continuous_symbol"]],
        schema=SCHEMA_1M, start=start, end=end, stype_in=STYPE,
    )
    log.info("Databento cost estimate for %s %s 1m: $%.4f", sym, start[:7]+"/"+end[:7], cost)

    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[INSTRUMENTS[sym]["continuous_symbol"]],
        schema=SCHEMA_1M, start=start, end=end, stype_in=STYPE,
    )
    df = data.to_df()
    df = df.reset_index() if df.index.name == "ts_event" else df
    df = df.rename(columns={"ts_event": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    wanted = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in wanted:
        if c not in df.columns:
            raise KeyError(f"Downloaded frame missing column: {c}")
    if "symbol" in df.columns:
        df = df.rename(columns={"symbol": "_contract"})
    elif "_contract" not in df.columns:
        df["_contract"] = INSTRUMENTS[sym]["continuous_symbol"]
    return df[wanted + ["_contract"]].copy()


def _compute_atr_4h(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    """
    4-hour Wilder ATR anchored at 18:00 UTC (≈18:00 ET in EDT season).
    Returns an array aligned to df's rows: each bar receives the ATR of the
    most recently COMPLETED 4h bar (shift-1, no lookahead).
    NaN where history is insufficient.
    """
    anchor_h = ATR_ANCHOR_ET[0]   # 18
    ts = df["timestamp"]
    ohlc_4h = (
        df.assign(_ts=ts)
        .set_index("_ts")
        .resample(f"{ATR_BAR_MINUTES}min", offset=f"{anchor_h}h")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"))
        .dropna(subset=["open"])
    )
    prev_c = ohlc_4h["close"].shift(1)
    tr = pd.concat([
        ohlc_4h["high"] - ohlc_4h["low"],
        (ohlc_4h["high"] - prev_c).abs(),
        (ohlc_4h["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr_prev = atr.shift(1)  # no lookahead

    atr_df = (
        atr_prev.reset_index()
        .rename(columns={"_ts": "timestamp", 0: "atr_4h"})
        .dropna(subset=["atr_4h"])
    )
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        atr_df.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    return merged["atr_4h"].to_numpy(copy=True)


def _compute_enrichment(df_1m: pd.DataFrame, df_1d: pd.DataFrame,
                        tick_size: float = 0.25) -> pd.DataFrame:
    """Add per-bar enrichment columns derived from 1d data and rolling 1m stats."""
    # ── daily enrichment (joined via ET date) ──────────────────────────────
    et = df_1d["timestamp"].dt.tz_convert("America/New_York") if df_1d is not None and len(df_1d) else None
    if et is not None:
        d = df_1d.copy()
        d["_date_et"] = et.dt.date
        d = d.sort_values("_date_et")
        d["prev_close"]     = d["close"].shift(1)
        d["daily_range_ticks"] = (d["high"] - d["low"]) / tick_size
        # Parkinson estimator: Var = (ln H/L)^2 / (4 ln 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_hl = np.log(d["high"] / d["low"])
        d["parkinson_var"]  = log_hl**2 / (4 * np.log(2))
        d["parkinson_vol_14d"] = (
            d["parkinson_var"].rolling(14, min_periods=7).mean()**0.5
            * np.sqrt(252)
        )
        # Realized vol from daily log-returns
        d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
        d["realized_vol_14d"] = (
            d["log_ret"].rolling(14, min_periods=7).std() * np.sqrt(252)
        )
        daily_cols = d[["_date_et", "prev_close", "daily_range_ticks",
                        "parkinson_vol_14d", "realized_vol_14d"]]
    else:
        daily_cols = None

    # ── join to 1m bars via ET date ────────────────────────────────────────
    df = df_1m.copy()
    df["_date_et"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    if daily_cols is not None:
        df = df.merge(daily_cols, on="_date_et", how="left")
    else:
        for col in ("prev_close", "daily_range_ticks", "parkinson_vol_14d", "realized_vol_14d"):
            df[col] = np.nan
    df = df.drop(columns=["_date_et"])
    return df


def _merge_and_cache(sym: str, existing: pd.DataFrame,
                     downloaded: pd.DataFrame, api_key: str | None) -> pd.DataFrame:
    """Merge existing + downloaded frames, compute ATR, save cache."""
    frames = [f for f in [downloaded, existing] if f is not None and len(f) > 0]
    if not frames:
        raise RuntimeError(f"No data available for {sym}.")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    log.info("%s merged: %d bars %s → %s", sym, len(df), df.timestamp.min(), df.timestamp.max())

    # ATR
    df["atr_4h"] = _compute_atr_4h(df)

    # Save
    path = _cache_path(sym, SCHEMA_1M)
    df.to_parquet(path, index=False)
    log.info("Cached %s → %s", sym, path)
    return df


# ── Public API ─────────────────────────────────────────────────────────────

def ensure_data(sym: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Return merged, enriched 1-minute DataFrame for sym.
    Loads from cache if available; builds cache on first call.
    api_key: Databento key; only needed if cache is missing.

    Two download paths:
      has_local_data=True  (ES, NQ): download 2019-2022, merge with local 2023-2026.
      has_local_data=False (all new): download data_start → today from Databento only.
    """
    cache = _cache_path(sym, SCHEMA_1M)
    if cache.exists():
        log.info("Loading %s from cache: %s", sym, cache)
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    if api_key is None:
        api_key = os.environ.get("DATABENTO_API_KEY")
    if api_key is None:
        raise ValueError("api_key required to build cache. "
                         "Set DATABENTO_API_KEY env var or pass explicitly.")

    instr = INSTRUMENTS[sym]
    if instr.get("has_local_data", False):
        # Legacy path: Databento 2019-2022 + local Dukascopy 2023-2026
        existing   = _load_existing_1m(sym)
        downloaded = _download_databento(sym, DOWNLOAD_START, DOWNLOAD_END, api_key)
        return _merge_and_cache(sym, existing, downloaded, api_key)
    else:
        # New instruments: no local backup — download full history from Databento
        start = instr.get("data_start", DOWNLOAD_START)
        end   = safe_end_date()
        log.info("%s: no local data — downloading %s → %s from Databento", sym, start, end)
        downloaded = _download_databento(sym, start, end, api_key)
        return _merge_and_cache(sym, pd.DataFrame(), downloaded, api_key)


def ensure_daily(sym: str, api_key: str | None = None) -> pd.DataFrame:
    """Return 1-day OHLCV for sym. Pulled from Databento if cache absent."""
    cache = _cache_path(sym, SCHEMA_1D)
    if cache.exists():
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    if api_key is None:
        api_key = os.environ.get("DATABENTO_API_KEY")
    if api_key is None:
        raise ValueError("api_key required.")

    try:
        import databento as db
    except ImportError:
        raise ImportError("pip install databento")

    client = db.Historical(key=api_key)
    # Pull from per-instrument start to the licence-safe end for enrichment
    start = INSTRUMENTS[sym].get("data_start", DOWNLOAD_START)
    end   = safe_end_date()
    data  = client.timeseries.get_range(
        dataset=DATASET, symbols=[INSTRUMENTS[sym]["continuous_symbol"]],
        schema=SCHEMA_1D, start=start, end=end, stype_in=STYPE,
    )
    df = data.to_df().reset_index()
    df = df.rename(columns={"ts_event": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(cache, index=False)
    log.info("Cached daily %s → %s", sym, cache)
    return df
