"""
Regression tests for regime-window placement.

The bug these cover: `select_windows` computed eligible_months as
`len(period_range(first_month, eligible_end, freq="M"))`, which counts the
BOUNDARY MONTH as fully usable even though only its first day(s) fall inside the
eligible span. Window ends are `start + REGIME_WINDOW_MONTHS - 1 day` and were
never clamped against `eligible_end`, and the only post-hoc assertion checked
windows for mutual overlap -- not for crossing the holdout.

Consequence: when the arithmetic left zero spare months, the final fitted window
ran past the holdout boundary and fitted-window trades leaked into the
out-of-sample region. Observed in production as ~6.2k ETH trades dated after the
holdout start appearing in trade_log.parquet.

Why it survived: the failure is DATA-DEPENDENT. With more history the spare months
absorb the miscount and no breach appears, so it comes and goes as the sample grows
rather than failing consistently.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import (
    HOLDOUT_MONTHS, N_REGIMES, REGIME_SEED, REGIME_WINDOW_MONTHS,
)
from src.regime_sampler import select_windows


def _eligible_end(data_end: date) -> date:
    """The holdout boundary as select_windows derives it."""
    return pd.Timestamp(data_end - pd.DateOffset(months=HOLDOUT_MONTHS)).date()


def _legacy_select(data_start: date, data_end: date):
    """Pre-fix placement logic, kept so the test proves the bug was real.

    Mirrors the original implementation: sliding cutoff, boundary month counted
    whole, no clamp, no breach guard.
    """
    rng = np.random.default_rng(REGIME_SEED)
    eligible_end = _eligible_end(data_end)
    first_month = pd.Timestamp(data_start) + pd.offsets.MonthBegin(0)
    if first_month.date() < data_start:
        first_month += pd.offsets.MonthBegin(1)
    eligible_months = len(pd.period_range(first_month, eligible_end, freq="M"))
    realised_n = min(N_REGIMES, eligible_months // REGIME_WINDOW_MONTHS)
    if realised_n == 0:
        return []
    spare = eligible_months - realised_n * REGIME_WINDOW_MONTHS
    gaps = np.full(realised_n + 1, spare // (realised_n + 1), dtype=int)
    remainder = spare % (realised_n + 1)
    if remainder:
        gaps[rng.choice(len(gaps), size=remainder, replace=False)] += 1
    out = []
    cursor = first_month + pd.DateOffset(months=int(gaps[0]))
    for i in range(realised_n):
        start = cursor.date()
        end = (cursor + pd.DateOffset(months=REGIME_WINDOW_MONTHS)
               - pd.DateOffset(days=1)).date()
        out.append((i, start, end))
        cursor += pd.DateOffset(months=REGIME_WINDOW_MONTHS + int(gaps[i + 1]))
    return out


# The exact production case. ETH's data_start with the cache state that shipped
# the leak: eligible_end 2026-02-03, exactly 10 windows, zero slack.
ETH_START = date(2021, 2, 8)
ETH_DATA_END = date(2026, 5, 3)


def test_legacy_logic_did_breach_the_holdout():
    """Guards the guard: proves the bug was real, so this file cannot pass vacuously.

    If this ever fails, the reproduction has drifted and the tests below stop
    demonstrating anything.
    """
    eligible_end = _eligible_end(ETH_DATA_END)
    legacy = _legacy_select(ETH_START, ETH_DATA_END)
    breaches = [w for w in legacy if w[2] >= eligible_end]
    assert breaches, (
        "expected the pre-fix logic to breach the holdout for the ETH case; "
        "reproduction has drifted")
    # 2025-09-01 -> 2026-02-28 against a 2026-02-03 boundary: 25 days over.
    worst = max((w[2] - eligible_end).days for w in breaches)
    assert worst >= 20, f"expected a multi-week breach, got {worst} days"


def test_no_window_crosses_holdout_boundary_eth_case():
    """The fix, on the exact case that leaked ~6.2k trades into the holdout."""
    eligible_end = _eligible_end(ETH_DATA_END)
    windows = select_windows(ETH_START, ETH_DATA_END)
    assert windows, "expected at least one window"
    for w in windows:
        assert w.end < eligible_end, (
            f"W{w.index:02d} {w.start}->{w.end} crosses holdout {eligible_end}")


@pytest.mark.parametrize("data_start", [
    date(2019, 1, 1), date(2019, 4, 1), date(2021, 2, 8), date(2022, 7, 15),
])
@pytest.mark.parametrize("data_end", [
    date(2025, 12, 31), date(2026, 2, 26), date(2026, 5, 3),
    date(2026, 8, 2), date(2026, 8, 7),
])
def test_no_window_crosses_holdout_boundary_grid(data_start, data_end):
    """The invariant must hold for every (start, end) pair, not just the one that
    happened to fail. The bug is data-dependent, so a single case is not coverage.
    """
    eligible_end = _eligible_end(data_end)
    windows = select_windows(data_start, data_end)
    for w in windows:
        assert w.end < eligible_end, (
            f"start={data_start} end={data_end}: W{w.index:02d} "
            f"{w.start}->{w.end} crosses holdout {eligible_end}")


@pytest.mark.parametrize("data_start,data_end", [
    (date(2019, 1, 1), date(2026, 5, 3)),
    (date(2021, 2, 8), date(2026, 5, 3)),
    (date(2019, 4, 1), date(2026, 8, 7)),
])
def test_windows_remain_non_overlapping_and_ordered(data_start, data_end):
    """The pre-existing invariant must not regress while fixing the new one."""
    windows = sorted(select_windows(data_start, data_end), key=lambda w: w.start)
    for a, b in zip(windows, windows[1:]):
        assert a.end < b.start, f"{a.start}->{a.end} overlaps {b.start}->{b.end}"
    for w in windows:
        assert w.start <= w.end


def test_window_length_is_preserved():
    """Clamping must not silently shorten windows -- it reduces the COUNT instead."""
    windows = select_windows(date(2019, 1, 1), date(2026, 5, 3))
    for w in windows:
        expected_end = (pd.Timestamp(w.start)
                        + pd.DateOffset(months=REGIME_WINDOW_MONTHS)
                        - pd.DateOffset(days=1)).date()
        assert w.end == expected_end, (
            f"W{w.index:02d} is not {REGIME_WINDOW_MONTHS} months: "
            f"{w.start}->{w.end}, expected end {expected_end}")


def test_insufficient_history_yields_fewer_windows_not_a_breach():
    """With too little history the correct response is fewer windows.

    Before the fix, a zero-slack span reached N_REGIMES by borrowing days from the
    holdout. The ETH case is exactly this: it yields N_REGIMES-1 windows now.
    """
    windows = select_windows(ETH_START, ETH_DATA_END)
    eligible_end = _eligible_end(ETH_DATA_END)
    assert len(windows) <= N_REGIMES
    assert all(w.end < eligible_end for w in windows)
    # Deliberately tight history: must degrade gracefully, never breach.
    tiny = select_windows(date(2025, 1, 1), date(2026, 5, 3))
    assert all(w.end < eligible_end for w in tiny)


def test_determinism_under_fixed_seed():
    """Placement is seeded; identical inputs must give identical windows."""
    a = select_windows(date(2019, 1, 1), date(2026, 5, 3))
    b = select_windows(date(2019, 1, 1), date(2026, 5, 3))
    assert [(w.index, w.start, w.end) for w in a] == \
           [(w.index, w.start, w.end) for w in b]


# ── holdout pin ────────────────────────────────────────────────────────────

def test_pin_defaults_to_none_so_behaviour_is_unchanged():
    """The pin must be opt-in. A default value would move every existing result."""
    from src.config import HOLDOUT_PIN_START
    assert HOLDOUT_PIN_START is None


def test_pin_fixes_the_boundary_across_data_extension():
    """The point of the pin: the same boundary regardless of how much data is loaded.

    Unpinned, extending data_end slides the holdout, so a result stops being
    comparable to the one before the extension -- and any period newly pulled into
    range silently enters the out-of-sample slice.
    """
    import importlib
    import src.config as cfg
    import src.regime_sampler as rs

    original = cfg.HOLDOUT_PIN_START
    try:
        cfg.HOLDOUT_PIN_START = "2026-02-01"
        importlib.reload(rs)
        short = rs.select_windows(date(2019, 1, 1), date(2026, 5, 3))
        long_ = rs.select_windows(date(2019, 1, 1), date(2026, 8, 7))
        assert [(w.start, w.end) for w in short] == [(w.start, w.end) for w in long_], (
            "pinned windows must not move when data_end advances")
        for w in long_:
            assert w.end < date(2026, 2, 1)
    finally:
        cfg.HOLDOUT_PIN_START = original
        importlib.reload(rs)


def test_unpinned_boundary_does_move_with_data_end():
    """Guards the guard: proves the pin is load-bearing rather than decorative."""
    short = select_windows(date(2019, 1, 1), date(2026, 5, 3))
    long_ = select_windows(date(2019, 1, 1), date(2026, 8, 7))
    assert [(w.start, w.end) for w in short] != [(w.start, w.end) for w in long_], (
        "unpinned windows should move with data_end; if not, the pin proves nothing")
