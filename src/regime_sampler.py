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


def assert_windows_within_data(windows: list[RegimeWindow], data_end: date,
                               label: str | None = None) -> None:
    """Every fitted window must lie inside the data it was fitted on.

    Independent of the holdout-boundary check in select_windows, which compares
    windows against eligible_end -- a value config can supply via HOLDOUT_PIN_START.
    If that value is wrong the boundary check validates windows against a fiction and
    passes. This compares against data_end, which is measured from the loaded data
    and cannot be asserted by config, so it holds even when the pin is nonsense.

    Currently unreachable through select_windows: the pin guard rejects
    pin >= data_end earlier, and unpinned runs derive eligible_end as
    data_end - HOLDOUT_MONTHS, which is strictly before data_end. It is kept as
    defence in depth and exposed as a named function so the invariant is testable
    directly -- asserting it through a reimplementation in the test would pass even
    if this check were deleted.
    """
    who = f"[{label}] " if label else ""
    outside = [w for w in windows if w.end > data_end]
    if outside:
        raise AssertionError(
            f"{who}Fitted window(s) extend past data_end {data_end}: "
            + ", ".join(f"W{w.index:02d} {w.start}->{w.end}" for w in outside)
            + ". Those windows are fitted on data that does not exist.")


def select_windows(data_start: date, data_end: date,
                   label: str | None = None) -> list[RegimeWindow]:
    """
    Given the span of available data, return N_REGIMES non-overlapping
    REGIME_WINDOW_MONTHS-month windows with even temporal spread.

    data_start, data_end: first and last trading dates available.
    label: optional caller identity (normally the symbol) used only in error text.
        This runs PER SYMBOL with that symbol's own data_end, so a pin can be valid
        for nine instruments and invalid for the tenth. Without the label that
        failure reports a bare date range and does not say which symbol produced it.
    """
    rng = np.random.default_rng(REGIME_SEED)
    who = f"[{label}] " if label else ""

    # Exclude holdout from the end.
    #
    # HOLDOUT_PIN_START, when set, fixes this boundary instead of deriving it from
    # data_end. The derived form is a function of how much data is loaded, so adding
    # data silently moves the out-of-sample slice -- and because this runs per symbol
    # with that symbol's own data_end, ragged caches give each symbol a DIFFERENT
    # boundary. None preserves the sliding default exactly.
    if HOLDOUT_PIN_START is not None:
        eligible_end = pd.Timestamp(HOLDOUT_PIN_START).date()
        # A pin is an ASSERTION about data that must already be loaded, so it has to
        # be checked against that data. Unvalidated, a pin at or after data_end
        # produced no holdout region at all and said nothing: windows were fitted
        # right up to the pin, the breach guard below compared them against the pin
        # itself and passed, and W09 landed beyond data_end entirely (verified:
        # pin 2027-01-01 with data_end 2026-04-30 placed W09 at 2026-04-01->2026-09-30).
        # Every downstream holdout read then covers an empty slice while every
        # artifact still claims an out-of-sample test was performed.
        #
        # Latent rather than live at the time of writing: the configured pin
        # 2026-02-01 sits 3-6 months before data_end for all 18 cached files. It
        # becomes live the moment a symbol's cache is rebuilt short, which is exactly
        # the ragged-cache scenario the pin exists to defend against.
        if eligible_end >= data_end:
            raise ValueError(
                f"{who}HOLDOUT_PIN_START={eligible_end} is at or after data_end "
                f"{data_end}, so no holdout region exists. Fitted windows would be "
                f"placed against a boundary outside the loaded data and the "
                f"out-of-sample test would silently cover nothing. Either load data "
                f"past {eligible_end} or set config.HOLDOUT_PIN_START to a date "
                f"before {data_end} (None restores the sliding boundary).")
        log.info("%sHoldout PINNED at %s (data_end=%s); sliding cutoff would have "
                 "been %s", who, eligible_end, data_end,
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
    # which counts the BOUNDARY MONTH as fully usable even though only its first
    # day(s) are. Window ends are start + REGIME_WINDOW_MONTHS - 1 day and were
    # never clamped against eligible_end, so when the arithmetic leaves no spare
    # months the final window ran past the holdout boundary and fitted-window
    # trades leaked into the out-of-sample region.
    #
    # Concrete case: data_start 2021-02-08, data_end 2026-05-03 gives
    # eligible_end 2026-02-03 and exactly 10 windows with zero slack, placing W09
    # at 2025-09-01 -> 2026-02-28 -- 25 days inside the holdout.
    #
    # The bug is data-dependent, which is why it survived: with more history the
    # spare months absorb the miscount and no breach appears, so it comes and goes
    # as the sample grows rather than failing consistently.
    # A month is usable only if its LAST DAY falls strictly before eligible_end.
    # Note (eligible_end - 1 day).to_period("M") is NOT sufficient: it is correct
    # only when eligible_end is the 1st of a month. For eligible_end 2026-02-03 it
    # returns 2026-02, whose month-end 2026-02-28 is already past the boundary.
    candidate = pd.Timestamp(eligible_end).to_period("M")
    last_month = (candidate - 1
                  if candidate.end_time.date() >= eligible_end else candidate)
    eligible_months = max(0, len(pd.period_range(
        first_month.to_period("M"), last_month, freq="M")))
    realised_n = min(N_REGIMES, eligible_months // REGIME_WINDOW_MONTHS)
    if realised_n < N_REGIMES:
        log.warning("%sHistory supports %d non-overlapping windows, requested %d",
                    who, realised_n, N_REGIMES)
    if realised_n == 0:
        # Genuinely-short history returning [] is legitimate (a symbol with a late
        # data_start simply has nowhere to fit windows). A PIN causing it is not: the
        # pin is a deliberate assertion, and silently producing zero fitted windows
        # means every variant for this symbol is scored on no data while the run
        # still reports success.
        if HOLDOUT_PIN_START is not None:
            raise ValueError(
                f"{who}HOLDOUT_PIN_START={eligible_end} leaves only "
                f"{eligible_months} usable month(s) after data_start {data_start}, "
                f"which fits zero {REGIME_WINDOW_MONTHS}-month windows. At least "
                f"{REGIME_WINDOW_MONTHS} are needed. Move the pin later or set it "
                f"to None.")
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

    # The guard that was missing. Mutual overlap was checked; crossing the holdout
    # boundary was not, so a fitted window could leak into the out-of-sample region
    # with no symptom other than holdout-dated rows appearing in the trade log --
    # which nothing asserted on either.
    breaches = [w for w in windows if w.end >= eligible_end]
    if breaches:
        raise AssertionError(
            f"{who}Fitted window(s) cross the holdout boundary {eligible_end}: "
            + ", ".join(f"W{w.index:02d} {w.start}->{w.end}" for w in breaches))

    assert_windows_within_data(windows, data_end, label=label)

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
