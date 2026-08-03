"""
test_entry_sm.py — unit tests for entry_detector state machines.

Uses minimal hand-crafted SessionDay objects to verify first-trigger logic
for each of the five entry modes (II, CC, TI, R-II, R-CC).
"""
import numpy as np
import pytest
from src.entry_detector import detect_entries, EntrySignal
from src.range_builder import SessionDay
from src.config import SESSIONS


def _sd(bars_ohlc, session="NY", instrument="ES", tick=0.25,
        rh=5400.0, rl=5380.0, rm=5):
    """
    Build a SessionDay where the active window starts at bar 0.
    bar_wall_mins are assigned starting at session_open + rm (range end).
    """
    arr = np.array(bars_ohlc, dtype=float)
    n = len(arr)
    sess = SESSIONS[session]
    oh, om = sess["open"]
    open_min = oh * 60 + om
    # Simulate bars starting right after range end
    wall = np.arange(open_min + rm, open_min + rm + n)   # bars start AFTER range end
    return SessionDay(
        instrument=instrument, session=session,
        local_date=None, local_tz=sess["tz"],
        bars_o=arr[:, 0], bars_h=arr[:, 1],
        bars_l=arr[:, 2], bars_c=arr[:, 3],
        bars_v=np.ones(n),
        bar_timestamps=np.zeros(n, dtype="int64"),
        bar_wall_mins=wall,
        range_highs={rm: rh, 15: rh, 30: rh},
        range_lows= {rm: rl, 15: rl, 30: rl},
        range_widths_ticks={rm: int((rh-rl)/tick), 15: int((rh-rl)/tick), 30: int((rh-rl)/tick)},
        atr_4h=100.0, tick_size=tick,
        prev_close=np.nan, gap_ticks=np.nan,
        parkinson_vol_14d=np.nan, realized_vol_14d=np.nan,
    )


class TestIIEntry:
    def test_long_ii_triggers_first_bar(self):
        # Open=5400.0 (below trigger), H=5401.0 crosses rh+tick=5400.25 → fill at trigger
        bars = [(5400.0, 5401.0, 5399.5, 5400.8)]
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "II" and s.direction == "long" and s.range_minutes == 5]
        assert len(sigs) == 1, f"expected 1 II long signal, got {len(sigs)}"
        s = sigs[0]
        assert s.fill_price == pytest.approx(5400.25, abs=0.01)
        assert s.gap_fill == False

    def test_short_ii_triggers_first_bar(self):
        # Open=5380.0 (above short trigger), L=5379.0 crosses rl-tick=5379.75 → fill at trigger
        bars = [(5380.0, 5380.5, 5379.0, 5379.8)]
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "II" and s.direction == "short" and s.range_minutes == 5]
        assert len(sigs) == 1
        assert sigs[0].fill_price == pytest.approx(5379.75, abs=0.01)
        assert sigs[0].gap_fill == False

    def test_no_ii_when_no_breakout(self):
        bars = [(5399.0, 5399.5, 5390.0, 5395.0)]  # H < 5400.25
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "II" and s.direction == "long"]
        assert len(sigs) == 0

    def test_gap_fill_flag(self):
        """Bar opening above trigger → gap_fill=True, fill at open."""
        bars = [(5402.0, 5405.0, 5401.0, 5403.0)]  # O > rh+tick
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "II" and s.direction == "long" and s.range_minutes == 5]
        assert len(sigs) == 1
        assert sigs[0].gap_fill == True
        assert sigs[0].fill_price == pytest.approx(5402.0, abs=0.01)


class TestCCEntry:
    def test_cc_1m_long(self):
        """1m close above boundary triggers CC."""
        bars = [
            (5399.0, 5399.5, 5398.0, 5399.0),   # bar0: close < rh — no trigger
            (5399.0, 5401.0, 5399.0, 5401.0),   # bar1: close > rh — CC trigger
        ]
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "CC" and s.closure_tf == 1 and s.direction == "long" and s.range_minutes == 5]
        assert len(sigs) == 1
        assert sigs[0].entry_bar_idx == 1
        assert sigs[0].fill_price == pytest.approx(5401.0, abs=0.01)


class TestTIEntry:
    def test_tap_in_after_breakout(self):
        """Price breaks out (bar0), retraces to boundary (bar1) → TI entry at bar1."""
        bars = [
            (5400.5, 5405.0, 5400.0, 5403.0),   # bar0: II breakout
            (5403.0, 5403.5, 5399.0, 5401.0),   # bar1: L≤5400 → tap-in
        ]
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "TI" and s.direction == "long" and s.range_minutes == 5]
        assert len(sigs) == 1
        ti = sigs[0]
        assert ti.entry_bar_idx == 1
        assert ti.fill_price == pytest.approx(5400.0, abs=0.01)


class TestRIIEntry:
    def test_r_ii_after_tap_in(self):
        """Breakout bar0, tap-in bar1, R-II bar2."""
        bars = [
            (5400.5, 5406.0, 5400.0, 5404.0),   # bar0: breakout
            (5404.0, 5404.5, 5399.0, 5401.0),   # bar1: tap-in (L≤5400)
            (5401.0, 5401.5, 5400.5, 5401.0),   # bar2: R-II (H≥5400.25)
        ]
        sd = _sd(bars)
        sigs = [s for s in detect_entries(sd) if s.mode == "R-II" and s.direction == "long" and s.range_minutes == 5]
        assert len(sigs) == 1
        assert sigs[0].entry_bar_idx == 2
        assert sigs[0].tap_in_bar_idx == 1
