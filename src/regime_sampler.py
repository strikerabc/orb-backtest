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
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    HOLDOUT_MONTHS, HOLDOUT_PIN_START, N_REGIMES, REGIME_SEED,
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

    # Exclude holdout from the end.
    #
    # HOLDOUT_PIN_START exists because this cutoff is RELATIVE to data_end, so
    # extending the sample silently slides the holdout forward. When the data was
    # extended to 2026-08-08, the sliding cutoff would have moved the holdout to
    # 2026-05-08 -> 2026-08-08, which SWALLOWS the paper-trading period
    # (2026-07-25 -> 2026-08-07) that generated the event-regime hypothesis. The
    # holdout would then contain its own originating observation, re-importing the
    # circularity EVENT_REGIME_PLAN.md section 1.2 exists to prevent -- and it
    # would do so invisibly, as a side effect of adding data.
    #
    # Pinning keeps the established 2026-02-01 boundary fixed, so the holdout that
    # produced "NO EDGE ESTABLISHED" stays the same slice of history and the newly
    # added May-July span becomes a SECOND, independent out-of-sample window.
    if HOLDOUT_PIN_START is not None:
        eligible_end = pd.Timestamp(HOLDOUT_PIN_START).date()
        log.info("Holdout PINNED at %s (data_end=%s); sliding cutoff would have "
                 "been %s", eligible_end, data_end,
                 pd.Timestamp(data_end - pd.DateOffset(months=HOLDOUT_MONTHS)).date())
    else:
        holdout_cutoff = data_end - pd.DateOffset(months=HOLDOUT_MONTHS)
        eligible_end   = pd.Timestamp(holdout_cutoff).date()

    first_month = pd.Timestamp(data_start) + pd.offsets.MonthBegin(0)
    if first_month.date() < data_start:
        first_month += pd.offsets.MonthBegin(1)

    # Count only months that end STRICTLY BEFORE eligible_end.
    #
    # The previous form was len(period_range(first_month, eligible_end, freq="M")),
    # which counts the boundary month as fully usable. With eligible_end
    # 2026-02-01 that credited all of February, so ETH -- data_start 2021-02-08,
    # exactly 10 windows in 61 counted months, zero slack -- placed W9 at
    # 2025-09-01 -> 2026-02-28, crossing the holdout boundary by 27 days and
    # putting 6,168 fitted trades inside the holdout region.
    #
    # Window ends are start + REGIME_WINDOW_MONTHS - 1 day and were never clamped
    # against eligible_end, and the only post-hoc check was for mutual overlap.
    # A window may share the boundary month only if it ends before the boundary.
    last_month = (pd.Timestamp(eligible_end) - pd.Timedelta(days=1)).to_period("M")
    eligible_months = len(pd.period_range(
        first_month.to_period("M"), last_month, freq="M"))
    realised_n = min(N_REGIMES, eligible_months // REGIME_WINDOW_MONTHS)
    if realised_n < N_REGIMES:
        log.warning("History supports %d non-overlapping windows, requested %d",
                    realised_n, N_REGIMES)
    if realised_n == 0:
        return []

    # Spread spare months almost evenly across the n+1 gaps. Randomly assign
    # the remainder so placement remains seeded without permitting overlap.
    spare = eligible_months - realised_n * REGIME_WINDOW_MONTHS
    gaps = np.full(realised_n + 1, spare // (realised_n + 1), dtype=int)
    remainder = spare % (realised_n + 1)
    if remainder:
        gaps[rng.choice(len(gaps), size=remainder, replace=False)] += 1

    windows: list[RegimeWindow] = []
    cursor = first_month + pd.DateOffset(months=int(gaps[0]))
    for i in range(realised_n):
        win_start = cursor.date()
        win_end = (cursor + pd.DateOffset(months=REGIME_WINDOW_MONTHS)
                   - pd.DateOffset(days=1)).date()
        windows.append(RegimeWindow(index=i, start=win_start, end=win_end))
        cursor += pd.DateOffset(
            months=REGIME_WINDOW_MONTHS + int(gaps[i + 1]))

    ordered = sorted(windows, key=lambda w: w.start)
    if any(a.end >= b.start for a, b in zip(ordered, ordered[1:])):
        raise AssertionError("Regime windows overlap")

    # The guard that was missing. Overlap was checked; crossing the holdout was
    # not, so a fitted window could leak into the out-of-sample region silently
    # and the only symptom was holdout-dated rows in the trade log.
    breaches = [w for w in windows if w.end >= eligible_end]
    if breaches:
        raise AssertionError(
            f"Fitted window(s) cross the holdout boundary {eligible_end}: "
            + ", ".join(f"W{w.index:02d} {w.start}->{w.end}" for w in breaches))

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
