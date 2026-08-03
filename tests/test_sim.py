"""
test_sim.py — unit tests for trade_sim exit-walk oracle.

Each test constructs a synthetic SessionDay + EntrySignal and verifies
that the exit reason, price, and R value are computed correctly.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.trade_sim import simulate_trade, TradeResult
from src.entry_detector import EntrySignal
from src.range_builder import SessionDay


def _make_sd(bars_ohlc, instrument="ES", session="NY",
             tick=0.25, atr=100.0, rh=5400.0, rl=5380.0):
    """Build a minimal SessionDay from a list of (O,H,L,C) tuples."""
    arr = np.array(bars_ohlc, dtype=float)
    n = len(arr)
    return SessionDay(
        instrument=instrument, session=session,
        local_date=None, local_tz="America/New_York",
        bars_o=arr[:, 0], bars_h=arr[:, 1],
        bars_l=arr[:, 2], bars_c=arr[:, 3],
        bars_v=np.ones(n), bar_timestamps=np.zeros(n, dtype="int64"),
        bar_wall_mins=np.arange(570, 570 + n),
        range_highs={5: rh, 15: rh, 30: rh},
        range_lows={5: rl, 15: rl, 30: rl},
        range_widths_ticks={5: int((rh - rl) / tick),
                            15: int((rh - rl) / tick),
                            30: int((rh - rl) / tick)},
        atr_4h=atr, tick_size=tick,
        prev_close=np.nan, gap_ticks=np.nan,
        parkinson_vol_14d=np.nan, realized_vol_14d=np.nan,
    )


def _make_es(entry_idx=0, fill=5401.0, sl=5381.0, rh=5400.0, rl=5380.0):
    return EntrySignal(
        mode="II", closure_tf=1, range_minutes=5,
        direction="long", entry_bar_idx=entry_idx,
        fill_price=fill, breakout_bar_idx=entry_idx,
        tap_in_bar_idx=None, boundary=rh,
        sl_price=sl, sl_bars_back=2, sl_source="cluster",
        gap_fill=False,
    )


class TestTPHit:
    def test_tp_hit_on_second_bar(self):
        # Entry at5401, SL at5381 → R=20 ticks. RR1.0 TP=5421.
        # Bar0: range (no SL/TP), Bar1: H=5422 → TP hit
        bars = [(5401, 5410, 5400, 5408),   # bar0 entry bar — H<TP, L>SL
                (5408, 5425, 5405, 5420)]   # bar1 — H≥5421
        sd = _make_sd(bars); es = _make_es()
        results = simulate_trade(es, sd, rr_levels=[1.0])
        r = results[0]
        assert r.exit_reason == "TP", f"Got {r.exit_reason}"
        assert r.exit_bar_idx == 1
        assert r.gross_r == pytest.approx(1.0, abs=0.01)


class TestSLHit:
    def test_sl_hit_on_first_bar(self):
        # L=5379 < SL=5381 → SL on bar0
        bars = [(5401, 5410, 5379, 5395)]
        sd = _make_sd(bars); es = _make_es()
        results = simulate_trade(es, sd, rr_levels=[1.0])
        r = results[0]
        assert r.exit_reason == "SL"
        assert r.exit_price == pytest.approx(5381.0, abs=0.01)
        assert r.gross_r == pytest.approx(-1.0, abs=0.1)


class TestTimeExit:
    def test_time_exit_last_bar(self):
        # Neither SL nor TP hit in 3 bars → time exit at close of last bar
        bars = [(5401, 5415, 5395, 5410),
                (5410, 5418, 5400, 5412),
                (5412, 5416, 5405, 5413)]   # 11:59 bar
        sd = _make_sd(bars); es = _make_es(sl=5350.0)  # SL far away
        results = simulate_trade(es, sd, rr_levels=[5.0])  # TP also far
        r = results[0]
        assert r.exit_reason == "TIME"
        assert r.exit_price == pytest.approx(5413.0, abs=0.01)


class TestSameBarAmbiguity:
    def test_same_bar_sl_first(self):
        # H=5422 (≥TP at RR1.0), L=5379 (≤SL) in same bar → SL first
        bars = [(5401, 5422, 5379, 5410)]
        sd = _make_sd(bars); es = _make_es()
        results = simulate_trade(es, sd, rr_levels=[1.0])
        r = results[0]
        assert r.exit_reason == "SL"
        assert r.same_bar_ambiguous is True


class TestGapFillSL:
    def test_gap_open_below_sl_fills_at_open(self):
        # Bar opens at 5375 which is below SL=5381 → fill at open
        bars = [(5375, 5380, 5370, 5378)]
        sd = _make_sd(bars); es = _make_es()
        results = simulate_trade(es, sd, rr_levels=[1.0])
        r = results[0]
        assert r.exit_reason == "SL"
        assert r.exit_price == pytest.approx(5375.0, abs=0.01)


class TestATRCap:
    def test_atr_exceeds_cap_flag(self):
        # TP at RR2 = 5401 + 2*20 = 5441. ATR=100. TP dist=40. 40/100=0.4 < 2.5 → NOT exceeded.
        bars = [(5401, 5445, 5399, 5440)]
        sd = _make_sd(bars, atr=100.0); es = _make_es()
        r2 = simulate_trade(es, sd, rr_levels=[2.0])[0]
        assert r2.atr_exceeds_cap is False

    def test_atr_cap_flagged_when_over(self):
        # R=20 ticks=5.0 pts, RR2 TP dist=10pts. ATR=3.0. 10/3=3.3 > 2.5 → flag
        sd = _make_sd([(5401, 5415, 5399, 5412)], atr=3.0)
        es = _make_es(sl=5396.0)   # R=5pts
        r2 = simulate_trade(es, sd, rr_levels=[2.0])[0]
        assert r2.atr_exceeds_cap is True
