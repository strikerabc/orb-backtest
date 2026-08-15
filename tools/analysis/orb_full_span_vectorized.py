"""
orb_full_span_vectorized.py — Fully vectorized trailing stop optimization.

Pre-builds all trade windows once, then uses pure NumPy for all configurations.
No DataFrame filtering inside loops = 1000× faster.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.data_layer import ensure_data
from src.range_builder import build_session_days
from src.entry_detector import detect_entries

INSTRUMENT = "NQ"
SESSION = "NY"
RANGE_MINS = 5
CLOSURE_TF = 5

# Test grid
ATR_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
BREAKEVEN_RR_LEVELS = [0.2, 0.25, 0.3]
TRAIL_START_RR_LEVELS = [0.4, 0.5, 0.6]
BASE_RR_TARGETS = [0.5, 0.75, 1.0]


@dataclass
class Config:
    atr_mult: float
    breakeven_rr: float
    trail_start_rr: float
    base_rr: float

    def label(self) -> str:
        return f"atr{self.atr_mult}_be{self.breakeven_rr}_ts{self.trail_start_rr}_rr{self.base_rr}"


class TradeWindow:
    """Pre-sliced bar arrays for one trade - built once, reused for all configs."""
    __slots__ = ('entry_price', 'stop_price', 'is_long', 'risk', 'atr', 'year',
                 'highs', 'lows', 'close_last')

    def __init__(self, entry_price, stop_price, is_long, risk, atr, year,
                 highs, lows, close_last):
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.is_long = is_long
        self.risk = risk
        self.atr = atr
        self.year = year
        self.highs = highs
        self.lows = lows
        self.close_last = close_last


def simulate_one_trade(window: TradeWindow, config: Config) -> float:
    """Simulate one trade with pure NumPy - no DataFrame operations."""
    if len(window.highs) == 0:
        return 0.0

    entry_price = window.entry_price
    stop_initial = window.stop_price
    is_long = window.is_long
    risk = window.risk

    target_price = (entry_price + config.base_rr * risk) if is_long else (entry_price - config.base_rr * risk)
    trail_dist = config.atr_mult * window.atr

    stop_current = stop_initial
    target_active = True
    trailing_active = False

    highs = window.highs
    lows = window.lows

    for i in range(len(highs)):
        fav_p = highs[i] if is_long else lows[i]
        adv_p = lows[i] if is_long else highs[i]

        # Check stop
        if is_long:
            if adv_p <= stop_current:
                return (stop_current - entry_price) / risk
        else:
            if adv_p >= stop_current:
                return (entry_price - stop_current) / risk

        # Check target
        if target_active:
            if is_long:
                if fav_p >= target_price:
                    return (target_price - entry_price) / risk
            else:
                if fav_p <= target_price:
                    return (entry_price - target_price) / risk

        # Calculate R
        fav_r = (fav_p - entry_price) / risk if is_long else (entry_price - fav_p) / risk

        # Breakeven
        if fav_r >= config.breakeven_rr and stop_current == stop_initial:
            stop_current = entry_price
            target_active = False

        # Start trailing
        if fav_r >= config.trail_start_rr and not trailing_active:
            trailing_active = True

        # Trail
        if trailing_active:
            if is_long:
                new_stop = fav_p - trail_dist
                stop_current = max(stop_current, new_stop)
            else:
                new_stop = fav_p + trail_dist
                stop_current = min(stop_current, new_stop)

    # Exit at close
    return (window.close_last - entry_price) / risk if is_long else (entry_price - window.close_last) / risk


def test_config_vectorized(config: Config, windows: list[TradeWindow]) -> dict:
    """Test one configuration across all pre-built trade windows."""
    returns = np.array([simulate_one_trade(w, config) for w in windows], dtype=float)
    years = np.array([w.year for w in windows], dtype=int)

    mean_r = float(returns.mean())
    std_r = float(returns.std())
    sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0.0
    win_rate = float((returns > 0).mean())
    median_r = float(np.median(returns))

    # Year-by-year consistency
    unique_years = np.unique(years)
    yearly_means = np.array([returns[years == y].mean() for y in unique_years])
    consistency = float((yearly_means > 0).mean())

    return {
        "config": config.label(),
        "atr_mult": config.atr_mult,
        "breakeven_rr": config.breakeven_rr,
        "trail_start_rr": config.trail_start_rr,
        "base_rr": config.base_rr,
        "n_trades": len(returns),
        "mean_r": round(mean_r, 5),
        "std_r": round(std_r, 5),
        "sharpe": round(sharpe, 3),
        "win_rate": round(win_rate, 4),
        "median_r": round(median_r, 5),
        "consistency_pct": round(consistency, 3),
        "total_r": round(float(returns.sum()), 2),
    }


def main():
    t0 = time.perf_counter()
    print("=" * 100, flush=True)
    print("FULL-SPAN EXPLORATION (VECTORIZED): ATR Trailing Stops", flush=True)
    print("=" * 100, flush=True)
    print(f"  Testing {len(ATR_MULTIPLIERS) * len(BREAKEVEN_RR_LEVELS) * len(TRAIL_START_RR_LEVELS) * len(BASE_RR_TARGETS)} configurations", flush=True)

    print("\n1. Loading NQ data...", flush=True)
    df_1m = ensure_data(INSTRUMENT)
    data_start = df_1m["timestamp"].min()
    data_end = df_1m["timestamp"].max()
    print(f"   {len(df_1m):,} bars from {data_start.date()} to {data_end.date()}", flush=True)

    print("\n2. Building sessions...", flush=True)
    session_days = build_session_days(df_1m, INSTRUMENT, SESSION)
    print(f"   {len(session_days):,} sessions", flush=True)

    print("\n3. Detecting entries...", flush=True)
    all_signals = []
    signal_to_sd = {}
    for sd in session_days:
        signals = detect_entries(sd)
        for sig in signals:
            all_signals.append(sig)
            signal_to_sd[id(sig)] = sd

    target_signals = [
        s for s in all_signals
        if s.mode == "CC" and s.range_minutes == RANGE_MINS and s.closure_tf == CLOSURE_TF
    ]
    print(f"   {len(target_signals):,} CC/5/5 signals", flush=True)

    # Build entries
    entries_data = []
    for sig in target_signals:
        sd = signal_to_sd[id(sig)]
        if sig.entry_bar_idx >= len(sd.bar_timestamps):
            continue

        bar_ts = pd.Timestamp(sd.bar_timestamps[sig.entry_bar_idx], tz="UTC")
        entries_data.append({
            "timestamp": bar_ts,
            "trigger_price": sig.fill_price,
            "direction": sig.direction,
            "stop_price": sig.sl_price,
            "atr_4h": sd.atr_4h,
            "year": bar_ts.year,
        })

    entries_df = pd.DataFrame(entries_data)
    print(f"   {len(entries_df):,} trades", flush=True)

    print("\n4. Pre-building all trade windows (one-time cost)...", flush=True)
    t_window_start = time.perf_counter()

    # Create timestamp index for fast lookups
    df_1m_sorted = df_1m.sort_values("timestamp").reset_index(drop=True)
    df_1m_sorted["_idx"] = np.arange(len(df_1m_sorted))

    windows = []
    for idx, row in enumerate(entries_df.itertuples(), 1):
        # Filter for bars after entry (vectorized boolean indexing)
        entry_ts = row.timestamp
        mask = df_1m_sorted["timestamp"] > entry_ts
        start_idx = mask.idxmax() if mask.any() else len(df_1m_sorted)

        if start_idx >= len(df_1m_sorted):
            continue

        df_post = df_1m_sorted.iloc[start_idx:]
        if len(df_post) == 0:
            continue

        highs = df_post["high"].to_numpy()
        lows = df_post["low"].to_numpy()
        close_last = df_post.iloc[-1]["close"]

        risk = abs(row.trigger_price - row.stop_price)
        is_long = row.direction == "long"

        window = TradeWindow(
            entry_price=row.trigger_price,
            stop_price=row.stop_price,
            is_long=is_long,
            risk=risk,
            atr=row.atr_4h,
            year=row.year,
            highs=highs,
            lows=lows,
            close_last=close_last
        )
        windows.append(window)

        if idx % 500 == 0:
            print(f"   Built {idx}/{len(entries_df)} windows...", flush=True)

    t_window_elapsed = time.perf_counter() - t_window_start
    print(f"   {len(windows):,} windows built in {t_window_elapsed:.1f}s", flush=True)

    # Generate configs
    configs = []
    for atr in ATR_MULTIPLIERS:
        for be in BREAKEVEN_RR_LEVELS:
            for ts in TRAIL_START_RR_LEVELS:
                for rr in BASE_RR_TARGETS:
                    configs.append(Config(atr, be, ts, rr))

    print(f"\n5. Testing {len(configs)} configurations...", flush=True)

    results = []
    for i, config in enumerate(configs, 1):
        result = test_config_vectorized(config, windows)
        results.append(result)

        if i % 10 == 0 or i == len(configs):
            elapsed = time.perf_counter() - t0
            eta = (elapsed / i) * (len(configs) - i)
            print(f"   Completed {i}/{len(configs)} configs ({i/len(configs)*100:.1f}%) — {elapsed:.1f}s elapsed, ETA {eta:.0f}s", flush=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 100, flush=True)
    print("TOP 20 CONFIGURATIONS (by Sharpe Ratio)", flush=True)
    print("=" * 100, flush=True)
    print(results_df.head(20).to_string(index=False), flush=True)

    print("\n" + "=" * 100, flush=True)
    print("TOP 10 BY CONSISTENCY (% years positive)", flush=True)
    print("=" * 100, flush=True)
    top_consistent = results_df.nlargest(10, "consistency_pct")
    print(top_consistent.to_string(index=False), flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n   Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    output = {
        "data_span": {"start": str(data_start.date()), "end": str(data_end.date())},
        "n_trades": len(windows),
        "n_configs": len(configs),
        "results": results_df.to_dict(orient="records"),
        "top_sharpe": results_df.head(10).to_dict(orient="records"),
        "top_consistent": top_consistent.to_dict(orient="records"),
        "elapsed_s": round(elapsed, 1),
    }

    out_path = _ROOT / "outputs" / "orb_full_span_exploration.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n   Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
