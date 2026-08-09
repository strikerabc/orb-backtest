"""
Tests for the optional hold cap on the exit walk.

Added for HYP-04's H4b, which asks whether the 11:59 exit is materially wrong. The
spec claims that test "requires no re-run"; it does, because mfe_r/mae_r are
TERMINAL aggregates over the whole hold and cannot reconstruct where price sat at
t=45min, and _first_touch_vectorized has no horizon parameter.

The load-bearing test here is not that the cap works -- it is that
`max_hold_bars=None` is byte-identical to the pre-change behaviour. This parameter
sits directly in the live sweep path, and a cap that leaked into production by
default would silently change every number in summary.parquet while looking like a
new feature working correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.entry_detector import EntrySignal
from src.trade_sim import simulate_trade, _simulate_trade


class FakeSD:
    """Minimal SessionDay stand-in. Only the fields _simulate_trade reads."""

    def __init__(self, o, h, l, c, *, instrument="ES", session="NY",
                 tick_size=0.25, atr_4h=10.0, session_open_idx=0):
        self.bars_o = np.asarray(o, dtype=float)
        self.bars_h = np.asarray(h, dtype=float)
        self.bars_l = np.asarray(l, dtype=float)
        self.bars_c = np.asarray(c, dtype=float)
        self.instrument = instrument
        self.session = session
        self.tick_size = tick_size
        self.atr_4h = atr_4h
        self.session_open_idx = session_open_idx
        self.contract_changed_in_session = False
        self.contract_changed_since_prev_session = False
        self.session_bar_completeness = 1.0


def _signal(entry=100.0, sl=99.0, direction="long", idx=0, at_close=False):
    return EntrySignal(
        mode="CC", closure_tf=5, range_minutes=5, direction=direction,
        entry_bar_idx=idx, fill_price=entry, breakout_bar_idx=idx,
        tap_in_bar_idx=None, boundary=entry, sl_price=sl, sl_bars_back=1,
        sl_source="test", gap_fill=False, fill_at_bar_close=at_close,
    )


def _drift(n, start=100.0, step=0.001):
    """A path that never reaches the 1.0-wide TP or SL used by _signal(), so every
    exit is TIME and the cap alone decides the exit bar.

    The step is deliberately sub-tick (tick_size is 0.25). It has to be: the longest
    window here is 200 bars and the target sits 1.0 away, so any step at or above
    0.005 would reach it and turn a TIME exit into a TP. An earlier version used
    0.01, which drifted into the target at bar 101 and made
    test_none_is_the_incumbent_behaviour fail against a correct simulation.

    Sub-tick drift is not a realistic price path, and is not meant to be -- these
    tests exercise the truncation arithmetic. The drift exists only so consecutive
    closes differ, which is what lets the exit price be attributed to a specific bar.
    """
    c = start + step * np.arange(n)
    return c, c + 0.02, c - 0.02, c


# ── the regression that matters ────────────────────────────────────────────────

def test_none_is_the_incumbent_behaviour():
    """No cap must hold to the last bar available."""
    o, h, l, c = _drift(120)
    sd = FakeSD(o, h, l, c)
    res = simulate_trade(_signal(), sd, [1.0])
    assert res[0].bars_held == 120
    assert res[0].exit_reason == "TIME"


def test_default_call_is_identical_to_explicit_none():
    o, h, l, c = _drift(90)
    sd = FakeSD(o, h, l, c)
    a = simulate_trade(_signal(), sd, [0.5, 1.0, 2.0])
    b = simulate_trade(_signal(), sd, [0.5, 1.0, 2.0], max_hold_bars=None)
    for x, y in zip(a, b):
        assert x == y


def test_cap_at_or_beyond_available_bars_changes_nothing():
    """Guards the boundary. A cap wider than the window must not truncate, and must
    not produce an off-by-one against the uncapped result."""
    o, h, l, c = _drift(50)
    sd = FakeSD(o, h, l, c)
    base = simulate_trade(_signal(), sd, [1.0])[0]
    for cap in (50, 51, 500):
        got = simulate_trade(_signal(), sd, [1.0], max_hold_bars=cap)[0]
        assert got == base, f"cap={cap} changed an untruncated result"


def test_max_hold_bars_is_keyword_only():
    """A positional third argument is rr_levels. If the cap were positional, an
    existing 4-argument call could acquire one silently."""
    o, h, l, c = _drift(30)
    sd = FakeSD(o, h, l, c)
    with pytest.raises(TypeError):
        simulate_trade(_signal(), sd, [1.0], 10)          # type: ignore[misc]


# ── the cap itself ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cap", [1, 5, 15, 30, 45, 60])
def test_cap_forces_a_time_exit_at_exactly_n_bars(cap):
    o, h, l, c = _drift(200)
    sd = FakeSD(o, h, l, c)
    r = simulate_trade(_signal(), sd, [1.0], max_hold_bars=cap)[0]
    assert r.bars_held == cap
    assert r.exit_reason == "TIME"


def test_capped_exit_price_is_the_close_of_the_last_held_bar():
    o, h, l, c = _drift(200)
    sd = FakeSD(o, h, l, c)
    cap = 20
    r = simulate_trade(_signal(), sd, [1.0], max_hold_bars=cap)[0]
    assert r.exit_price == pytest.approx(c[cap - 1])


def test_cap_does_not_prevent_an_earlier_tp():
    """A TP inside the cap must still fire. Otherwise the cap would not be measuring
    a shorter hold, it would be suppressing exits."""
    n = 100
    c = np.full(n, 100.0)
    c[10:] = 101.5                       # crosses a 1.0 target at bar 10
    h = c + 0.02
    l = c - 0.02
    o = np.full(n, 100.0)
    o[10:] = 101.5
    sd = FakeSD(o, h, l, c)
    r = simulate_trade(_signal(), sd, [1.0], max_hold_bars=60)[0]
    assert r.exit_reason == "TP"
    assert r.bars_held <= 11


def test_cap_can_convert_a_tp_into_a_time_exit():
    """The whole point of H4b: a target reached late is unreachable under a cap."""
    n = 100
    c = np.full(n, 100.0)
    c[70:] = 101.5                       # target only reached at bar 70
    h = c + 0.02
    l = c - 0.02
    o = np.full(n, 100.0)
    o[70:] = 101.5
    sd = FakeSD(o, h, l, c)
    uncapped = simulate_trade(_signal(), sd, [1.0])[0]
    capped = simulate_trade(_signal(), sd, [1.0], max_hold_bars=45)[0]
    assert uncapped.exit_reason == "TP"
    assert capped.exit_reason == "TIME"
    assert capped.gross_r < uncapped.gross_r


def test_cap_can_convert_an_sl_into_a_time_exit():
    """The converse, and the reason a shorter horizon is not free: a stop avoided is
    a loss avoided, so the cap must be able to help as well as hurt. Without this
    test the suite would only pin the pessimistic direction."""
    n = 100
    c = np.full(n, 100.2)
    c[70:] = 98.5                        # stop only hit at bar 70
    h = c + 0.02
    l = c - 0.02
    o = np.full(n, 100.2)
    o[70:] = 98.5
    sd = FakeSD(o, h, l, c)
    uncapped = simulate_trade(_signal(), sd, [1.0])[0]
    capped = simulate_trade(_signal(), sd, [1.0], max_hold_bars=45)[0]
    assert uncapped.exit_reason == "SL"
    assert capped.exit_reason == "TIME"
    assert capped.gross_r > uncapped.gross_r


# ── excursions must respect the truncated window ───────────────────────────────

def test_mfe_and_mae_are_computed_over_the_truncated_window_only():
    """mfe_r/mae_r describe what the trade EXPERIENCED. Under a cap they must not
    include excursions that happen after the trade would have been closed --
    otherwise the horizon analysis reads post-exit information."""
    n = 100
    c = np.full(n, 100.0)
    h = c + 0.05
    l = c - 0.05
    h[80] = 100.9                        # big favourable spike, after the cap
    l[80] = 99.1                         # big adverse spike, after the cap
    o = c.copy()
    sd = FakeSD(o, h, l, c)
    capped = simulate_trade(_signal(), sd, [2.0], max_hold_bars=45)[0]
    uncapped = simulate_trade(_signal(), sd, [2.0])[0]
    assert capped.mfe_r < uncapped.mfe_r
    assert capped.mae_r > uncapped.mae_r          # mae_r is <= 0
    assert capped.mfe_r == pytest.approx(0.05, abs=1e-9)


# ── degenerate caps ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cap", [0, -1, -10])
def test_non_positive_cap_yields_no_hold_bars(cap):
    """Zero bars held is not a trade. It must be flagged invalid rather than
    silently returning a zero-R result, which would enter the pool as a break-even
    trade and dilute every expectancy it touched."""
    o, h, l, c = _drift(50)
    sd = FakeSD(o, h, l, c)
    r = simulate_trade(_signal(), sd, [1.0], max_hold_bars=cap)[0]
    assert r.exit_reason == "NO_HOLD_BARS"
    assert np.isnan(r.gross_r)


def test_cap_interacts_correctly_with_fill_at_bar_close():
    """fill_at_bar_close shifts `start` forward by one, and the cap is measured from
    the fill, not from the signal bar."""
    o, h, l, c = _drift(200)
    sd = FakeSD(o, h, l, c)
    at_close = simulate_trade(_signal(idx=5, at_close=True), sd, [1.0],
                             max_hold_bars=30)[0]
    at_open = simulate_trade(_signal(idx=5, at_close=False), sd, [1.0],
                             max_hold_bars=30)[0]
    assert at_close.bars_held == 30
    assert at_open.bars_held == 30
    # Same number of bars held, but shifted one bar later in the session.
    assert at_close.exit_bar_idx == at_open.exit_bar_idx + 1


def test_cost_r_is_unchanged_by_the_cap():
    """A shorter hold does not change r_ticks, so cost_r must be identical. This is
    what lets H4b report the gross delta and the net delta as the same number."""
    o, h, l, c = _drift(200)
    sd = FakeSD(o, h, l, c)
    a = simulate_trade(_signal(), sd, [1.0])[0]
    b = simulate_trade(_signal(), sd, [1.0], max_hold_bars=45)[0]
    assert a.cost_r == pytest.approx(b.cost_r, rel=1e-12)
    assert a.r_ticks == pytest.approx(b.r_ticks, rel=1e-12)
    assert (a.gross_r - a.net_r) == pytest.approx(b.gross_r - b.net_r, abs=1e-9)


def test_reference_walker_honours_the_cap_too():
    """The vectorised and reference walkers must agree under a cap, or the parity
    check that guards the fast path would stop covering capped runs."""
    o, h, l, c = _drift(200)
    sd = FakeSD(o, h, l, c)
    fast = _simulate_trade(_signal(), sd, [1.0], reference=False, max_hold_bars=45)[0]
    slow = _simulate_trade(_signal(), sd, [1.0], reference=True, max_hold_bars=45)[0]
    assert fast.exit_reason == slow.exit_reason
    assert fast.bars_held == slow.bars_held
    assert fast.gross_r == pytest.approx(slow.gross_r, abs=1e-9)
