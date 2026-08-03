"""
swing_detector.py — locate the pre-breakout stop-loss level.

A5 (confirmed): only search for down-closing clusters in bars that occurred
BEFORE the initial breakout bar (exclusive). This prevents bars in a slow
retracement back to the range from being mistakenly selected as the SL cluster.

down-closing bar: close < open  (not close < prior_close)
SL level        : min(low) of the most recent maximal consecutive run
fallback        : range_low (long) / range_high (short), flagged
"""
from __future__ import annotations

import numpy as np

from src.config import SWING_MAX_LOOKBACK_BARS, SWING_MIN_SL_TICKS


def _find_cluster(opens: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                  search_end: int, max_lookback: int) -> tuple[int, int]:
    """
    Scan backward from search_end (exclusive) to find the most recent
    maximal run of consecutive down-closing bars.
    Returns (run_start_idx, run_end_idx) both inclusive, or (-1, -1) if none.
    """
    start = max(0, search_end - max_lookback)
    down  = closes[start:search_end] < opens[start:search_end]
    n     = len(down)
    if n == 0:
        return -1, -1

    # Find last down-close bar (scan from end)
    run_end = -1
    for i in range(n - 1, -1, -1):
        if down[i]:
            run_end = i
            break
    if run_end == -1:
        return -1, -1

    # Extend run backward while consecutive
    run_start = run_end
    while run_start > 0 and down[run_start - 1]:
        run_start -= 1

    return start + run_start, start + run_end


def find_swing_low(
    bars_o: np.ndarray, bars_h: np.ndarray,
    bars_l: np.ndarray, bars_c: np.ndarray,
    breakout_bar_idx: int,
    tick_size: float,
    range_low: float,
    max_lookback: int = SWING_MAX_LOOKBACK_BARS,
    min_sl_ticks: int = SWING_MIN_SL_TICKS,
) -> tuple[float, int, str]:
    """
    For a LONG trade: find SL below entry.
    Returns (sl_price, bars_back, source).
      source: 'cluster' | 'range_fallback' | 'min_floor_applied'
    """
    if breakout_bar_idx <= 0:
        sl = range_low - min_sl_ticks * tick_size
        return sl, 0, "no_prior_bars"

    rs, re = _find_cluster(bars_o, bars_l, bars_c,
                           breakout_bar_idx, max_lookback)
    if rs == -1:
        return range_low, 0, "range_fallback"

    sl      = float(bars_l[rs:re + 1].min())
    bars_bk = breakout_bar_idx - rs
    # Use cluster low as-is. Degenerate R (too small) is caught in trade_sim.
    return sl, bars_bk, "cluster"


def find_swing_high(
    bars_o: np.ndarray, bars_h: np.ndarray,
    bars_l: np.ndarray, bars_c: np.ndarray,
    breakout_bar_idx: int,
    tick_size: float,
    range_high: float,
    max_lookback: int = SWING_MAX_LOOKBACK_BARS,
    min_sl_ticks: int = SWING_MIN_SL_TICKS,
) -> tuple[float, int, str]:
    """
    For a SHORT trade: SL above entry (mirror of find_swing_low).
    A down-close cluster mirrors to an UP-close cluster for shorts.
    """
    if breakout_bar_idx <= 0:
        sl = range_high + min_sl_ticks * tick_size
        return sl, 0, "no_prior_bars"

    # For shorts, search for up-closing cluster (close > open) before breakout
    up   = bars_c[:breakout_bar_idx] > bars_o[:breakout_bar_idx]
    start = max(0, breakout_bar_idx - max_lookback)
    up_w  = bars_c[start:breakout_bar_idx] > bars_o[start:breakout_bar_idx]
    n = len(up_w)
    run_end = -1
    for i in range(n - 1, -1, -1):
        if up_w[i]:
            run_end = i
            break
    if run_end == -1:
        return range_high, 0, "range_fallback"

    run_start = run_end
    while run_start > 0 and up_w[run_start - 1]:
        run_start -= 1

    abs_rs = start + run_start
    abs_re = start + run_end
    sl      = float(bars_h[abs_rs:abs_re + 1].max())
    bars_bk = breakout_bar_idx - abs_rs
    return sl, bars_bk, "cluster"
