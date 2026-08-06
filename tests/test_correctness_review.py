from dataclasses import fields
from datetime import date

import numpy as np
import pandas as pd

from src.entry_detector import EntrySignal, _resample_to_tf
from src.filters import trade_eligibility
from src.null_calibrator import sample_null_days
from src.range_builder import SessionDay
from src.swing_detector import find_swing_high, find_swing_low
from src.trade_sim import _simulate_trade_reference, simulate_trade


def _session(bars, *, tick=0.25, session_open_idx=0):
    a = np.asarray(bars, dtype=float)
    n = len(a)
    return SessionDay(
        "ES", "NY", date(2024, 1, 2), "America/New_York",
        a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.ones(n),
        np.arange(n, dtype=np.int64), np.arange(570, 570 + n),
        {5: 100.0}, {5: 95.0}, {5: 20.0}, 20.0, tick,
        np.nan, np.nan, np.nan, np.nan,
        session_open_idx=session_open_idx,
    )


def _signal(**overrides):
    values = dict(
        mode="II", closure_tf=1, range_minutes=5, direction="long",
        entry_bar_idx=0, fill_price=101.0, breakout_bar_idx=0,
        tap_in_bar_idx=None, boundary=100.0, sl_price=99.0,
        sl_bars_back=0, sl_source="cluster", gap_fill=False,
        fill_at_bar_close=False,
    )
    values.update(overrides)
    return EntrySignal(**values)


def test_close_fill_does_not_exit_on_confirming_bar():
    sd = _session([
        (100, 104, 98, 101),
        (101, 101.5, 100.5, 101),
        (101, 101.5, 100.5, 101),
    ])
    result = simulate_trade(_signal(fill_at_bar_close=True), sd, [5.0])[0]
    assert result.exit_reason == "TIME"
    assert result.exit_bar_idx == 2
    assert result.bars_held == 2


def test_close_fill_with_no_remaining_bar_is_explicitly_invalid():
    sd = _session([(100, 104, 98, 101)])
    result = simulate_trade(_signal(fill_at_bar_close=True), sd, [1.0])[0]
    assert result.exit_reason == "NO_HOLD_BARS"


def test_wrong_side_stop_cannot_be_a_profitable_sl():
    sd = _session([(101, 103, 100, 102)])
    result = simulate_trade(_signal(sl_price=102.0), sd, [1.0])[0]
    assert result.exit_reason == "SL_WRONG_SIDE"
    assert np.isnan(result.gross_r)


def test_swing_floor_is_relative_to_entry_and_mirrored():
    o = np.array([100.0, 101.0])
    h = np.array([101.5, 103.0])
    l = np.array([100.5, 100.5])
    c = np.array([99.5, 102.0])
    sl, _, source = find_swing_low(
        o, h, l, c, 1, 0.25, 99.0, min_sl_ticks=4, entry_price=101.0)
    assert sl == 100.0
    assert source == "min_floor_applied"

    c_short = np.array([100.5, 99.0])
    sl, _, source = find_swing_high(
        o, h, l, c_short, 1, 0.25, 103.0,
        min_sl_ticks=4, entry_price=101.0)
    assert sl == 102.0
    assert source == "min_floor_applied"


def test_confirmation_resample_rejects_incomplete_bucket():
    closes = np.array([1.0, 2.0, 3.0, 4.0])
    wall = np.array([575, 576, 577, 578])  # missing bucket close at 579
    tf_close, last = _resample_to_tf(575, closes, wall, 5)
    assert np.isnan(tf_close[0])
    assert last[0] == -1


def test_trade_through_required_for_resting_tp():
    sd = _session([(101, 103.0, 100.5, 102.0)])
    # Entry 101, stop 99 => TP 103. A touch is insufficient; closes at TIME.
    result = simulate_trade(_signal(), sd, [1.0])[0]
    assert result.exit_reason == "TIME"


def test_eligibility_is_centralized_and_reports_each_rule():
    frame = pd.DataFrame({
        "instrument": ["ES", "ES"], "exit_reason": ["TP", "TP"],
        "tp_unfillable": [False, False], "tp_ticks": [8.0, 2.0],
        "r_ticks": [20.0, 4.0], "cost_r": [0.1, 0.2],
        "contracts": [1, 0], "session_bar_completeness": [1.0, 0.5],
        "contract_changed_in_session": [False, True],
        "contract_changed_since_prev_session": [False, False],
    })
    marked = trade_eligibility(frame)
    assert bool(marked.loc[0, "eligible"])
    assert not bool(marked.loc[1, "eligible"])
    assert bool(marked.loc[1, "excluded_small_tp"])
    assert bool(marked.loc[1, "excluded_roll_day"])
    assert bool(marked.loc[1, "excluded_incomplete_session"])


def test_stratified_null_sampling_is_deterministic_and_not_a_head_slice():
    days = []
    for i in range(20):
        sd = _session([(101, 102, 100, 101)])
        sd.local_date = date(2024, 1, i + 1)
        sd.regime_window = i // 10
        days.append(sd)
    a, status_a = sample_null_days(
        days, 8, np.random.default_rng(42), window_of=lambda sd: sd.regime_window)
    b, status_b = sample_null_days(
        days, 8, np.random.default_rng(42), window_of=lambda sd: sd.regime_window)
    assert status_a == status_b == "stratified"
    assert [x.local_date for x in a] == [x.local_date for x in b]
    assert [x.local_date for x in a] != [x.local_date for x in days[:8]]
    assert {x.regime_window for x in a} == {0, 1}


def test_vectorized_simulator_matches_reference_randomized():
    rng = np.random.default_rng(20260806)
    for _ in range(100_000):
        n = int(rng.integers(1, 25))
        close = 100 + np.cumsum(rng.normal(0, 0.4, n))
        open_ = np.r_[100.0, close[:-1]] + rng.normal(0, 0.1, n)
        high = np.maximum(open_, close) + rng.random(n)
        low = np.minimum(open_, close) - rng.random(n)
        sd = _session(np.column_stack([open_, high, low, close]))
        is_long = bool(rng.integers(0, 2))
        entry = float(open_[0])
        distance = float(rng.integers(4, 80)) * 0.25
        signal = _signal(
            direction="long" if is_long else "short", fill_price=entry,
            sl_price=entry - distance if is_long else entry + distance,
            fill_at_bar_close=bool(rng.integers(0, 2)),
        )
        actual = simulate_trade(signal, sd, [0.25, 0.5, 1.0, 2.0])
        expected = _simulate_trade_reference(signal, sd, [0.25, 0.5, 1.0, 2.0])
        for left, right in zip(actual, expected):
            for field in fields(left):
                x, y = getattr(left, field.name), getattr(right, field.name)
                if isinstance(x, float):
                    assert np.isclose(x, y, equal_nan=True), field.name
                else:
                    assert x == y, field.name
