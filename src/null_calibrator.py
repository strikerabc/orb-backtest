"""
null_calibrator.py — random-entry benchmark and bootstrap CI.

Runs the exact same simulation machinery with randomized entry times so
we can answer: "could this variant's expectancy arise by chance alone?"
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import BOOTSTRAP_N, BOOTSTRAP_BLOCK_SIZE_DAYS, RR_LEVELS
from src.range_builder import SessionDay
from src.swing_detector import find_swing_low, find_swing_high
from src.trade_sim import simulate_trade, TradeResult
from src.entry_detector import EntrySignal

log = logging.getLogger("orb.null")


def _random_entry_signal(sd: SessionDay, rm: int, direction: str,
                          rng: np.random.Generator) -> EntrySignal | None:
    """
    Generate a random EntrySignal within the session active window.
    Uses range_low as SL (same floor the strategy would use) for fair comparison.
    """
    if rm not in sd.range_highs:
        return None
    n = len(sd.bars_o)
    if n == 0:
        return None

    tick = sd.tick_size
    rh = sd.range_highs[rm]; rl = sd.range_lows[rm]
    boundary = rh if direction == "long" else rl
    is_long  = direction == "long"

    # Random bar index in active window (after range)
    from src.config import SESSIONS
    sess = SESSIONS[sd.session]
    oh, om = sess["open"]
    open_min = oh * 60 + om
    range_end = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if range_end >= n:
        return None

    entry_idx = int(rng.integers(range_end, n))
    fill  = float(sd.bars_o[entry_idx])
    sl, slb, sls = (find_swing_low(sd.bars_o, sd.bars_h, sd.bars_l, sd.bars_c,
                                    entry_idx, tick, rl)
                    if is_long else
                    find_swing_high(sd.bars_o, sd.bars_h, sd.bars_l, sd.bars_c,
                                    entry_idx, tick, rh))
    return EntrySignal(
        mode="NULL", closure_tf=1, range_minutes=rm,
        direction=direction, entry_bar_idx=entry_idx,
        fill_price=fill, breakout_bar_idx=entry_idx,
        tap_in_bar_idx=None, boundary=boundary,
        sl_price=sl, sl_bars_back=slb, sl_source=sls,
        gap_fill=False,
    )


def build_null_distribution(
    session_days: list[SessionDay],
    range_minutes: int,
    direction: str,
    rr: float,
    seed: int = 99,
) -> np.ndarray:
    """
    Generate gross_r distribution for random entries.
    Returns 1-D array of gross_r values (one per session-day, NaN if no valid entry).
    """
    rng     = np.random.default_rng(seed)
    results = []
    for sd in session_days:
        es = _random_entry_signal(sd, range_minutes, direction, rng)
        if es is None:
            continue
        trades = simulate_trade(es, sd, rr_levels=[rr])
        if trades and trades[0].exit_reason not in ("INVALID", None):
            results.append(trades[0].gross_r)
    return np.array(results, dtype=float)


def null_p_value(observed_expectancy: float, null_distribution: np.ndarray) -> float:
    """Fraction of null runs that beat observed_expectancy."""
    if len(null_distribution) == 0:
        return np.nan
    return float(np.mean(null_distribution >= observed_expectancy))


def enrich_summary_with_null(
    summary_df: pd.DataFrame,
    session_days_map: dict,   # (sym, session) -> list[SessionDay]
    n_null_samples: int = 500,
) -> pd.DataFrame:
    """Add null_expectancy_mean, null_expectancy_p95, null_p_value columns."""
    rows = []
    for _, row in summary_df.iterrows():
        sym  = row["instrument"]; sess = row["session"]
        rm   = int(row["range_minutes"]); rr = float(row["rr"])
        d    = row["direction"]
        sds  = session_days_map.get((sym, sess), [])
        null = build_null_distribution(sds[:n_null_samples], rm, d, rr)
        pv   = null_p_value(float(row.get("expectancy_gross_r", 0.0)), null)
        row  = row.copy()
        row["null_exp_mean"] = round(float(np.nanmean(null)), 4) if len(null) else np.nan
        row["null_exp_p95"]  = round(float(np.nanpercentile(null, 95)), 4) if len(null) else np.nan
        row["null_p_value"]  = round(pv, 4)
        rows.append(row)
    return pd.DataFrame(rows)
