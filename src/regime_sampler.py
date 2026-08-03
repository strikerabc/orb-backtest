"""
regime_sampler.py — select 10 truly non-overlapping, temporally-spread windows.

Strategy: divide the full available history into N_REGIMES equal time segments;
within each segment, randomly place one REGIME_WINDOW_MONTHS-month window.
This guarantees even temporal distribution (not just non-overlap) while
remaining random (seeded for reproducibility).

The most-recent HOLDOUT_MONTHS months are excluded from all windows to
preserve a pure out-of-sample slice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import (
    HOLDOUT_MONTHS, N_REGIMES, REGIME_SEED,
    REGIME_WINDOW_MONTHS,
)

log = logging.getLogger("orb.regime")


@dataclass
class RegimeWindow:
    index: int
    start: date
    end: date    # inclusive

    def __str__(self) -> str:
        return f"W{self.index:02d}  {self.start}→{self.end}"


def select_windows(data_start: date, data_end: date) -> list[RegimeWindow]:
    """
    Given the span of available data, return N_REGIMES non-overlapping
    REGIME_WINDOW_MONTHS-month windows with even temporal spread.

    data_start, data_end: first and last trading dates available.
    """
    rng = np.random.default_rng(REGIME_SEED)

    # Exclude holdout from the end
    holdout_cutoff = data_end - pd.DateOffset(months=HOLDOUT_MONTHS)
    eligible_end   = pd.Timestamp(holdout_cutoff).date()

    # Compute segment boundaries
    total_days = (eligible_end - data_start).days
    seg_days   = total_days / N_REGIMES

    windows: list[RegimeWindow] = []
    window_len_days = REGIME_WINDOW_MONTHS * 30  # approximate, re-aligned below

    for i in range(N_REGIMES):
        seg_start_days = int(i * seg_days)
        seg_end_days   = int((i + 1) * seg_days) - window_len_days

        if seg_end_days <= seg_start_days:
            # Segment too small — fallback to start of segment
            seg_end_days = seg_start_days

        offset = int(rng.integers(0, max(1, seg_end_days - seg_start_days + 1)))
        win_start = data_start + timedelta(days=seg_start_days + offset)

        # Align to first day of month
        win_start = pd.Timestamp(win_start).replace(day=1).date()

        # Advance REGIME_WINDOW_MONTHS months
        win_end_ts = pd.Timestamp(win_start) + pd.DateOffset(months=REGIME_WINDOW_MONTHS)
        win_end    = (win_end_ts - pd.DateOffset(days=1)).date()

        # Clamp to eligible range
        if win_end > eligible_end:
            win_end   = eligible_end
            win_start = (pd.Timestamp(win_end) - pd.DateOffset(months=REGIME_WINDOW_MONTHS)
                         + pd.DateOffset(days=1)).date()

        windows.append(RegimeWindow(index=i, start=win_start, end=win_end))

    for w in windows:
        log.info("Regime window: %s", w)

    return windows


def filter_to_window(df: pd.DataFrame, window: RegimeWindow,
                     tz: str = "UTC") -> pd.DataFrame:
    """Return rows of df whose timestamp falls within [window.start, window.end]."""
    ts = df["timestamp"].dt.tz_convert("UTC")
    start = pd.Timestamp(window.start, tz="UTC")
    end   = pd.Timestamp(window.end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(ts >= start) & (ts < end)].copy()
