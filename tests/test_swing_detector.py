"""
test_swing_detector.py — unit tests for swing_detector using hand-crafted bars.

Each test constructs a minimal bar sequence and asserts exact SL placement.
"""
import numpy as np
import pytest
from src.swing_detector import find_swing_low, find_swing_high


def _bars(rows):
    """rows: list of (open, high, low, close)"""
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


class TestFindSwingLow:
    def test_basic_cluster_before_breakout(self):
        # Bars 0-2: up, 3-4: down-close cluster, 5: breakout
        opens  = [100, 101, 102, 103, 101, 102]
        highs  = [101, 102, 103, 103, 101, 106]
        lows   = [99,  100, 101, 99,  98,  102]
        closes = [101, 102, 103, 100, 99,  105]  # bars 3,4 are down-close
        o, h, l, c = _bars(list(zip(opens, highs, lows, closes)))
        sl, bars_back, source = find_swing_low(o, h, l, c,
                                               breakout_bar_idx=5,
                                               tick_size=0.25,
                                               range_low=95.0,
                                               max_lookback=50, min_sl_ticks=0)
        assert source == "cluster", f"expected cluster, got {source}"
        assert sl == 98.0, f"expected 98.0, got {sl}"
        assert bars_back == 2  # cluster ends at bar4, starts at bar3; breakout at5 → 5-3=2

    def test_only_searches_pre_breakout(self):
        """Bars AFTER breakout_bar_idx must be ignored even if they form a cluster."""
        o, h, l, c = _bars([
            (100, 101, 99, 101),   # bar0: up-close
            (101, 106, 101, 105),  # bar1: breakout bar
            (105, 106, 102, 103),  # bar2: post-breakout down-close — MUST BE IGNORED
            (103, 104, 100, 101),  # bar3: post-breakout down-close — MUST BE IGNORED
        ])
        sl, bars_back, source = find_swing_low(o, h, l, c,
                                               breakout_bar_idx=1,
                                               tick_size=0.25,
                                               range_low=90.0,
                                               max_lookback=50, min_sl_ticks=0)
        # Only bar0 is pre-breakout and it is UP-close, so no cluster → range_fallback
        assert source == "range_fallback", f"expected range_fallback, got {source}"
        assert sl == 90.0

    def test_cluster_low_used_verbatim(self):
        """Cluster low is returned as-is; degenerate-R filtering belongs in trade_sim."""
        o, h, l, c = _bars([
            (100, 101, 97.0, 99.5),   # down-close bar, low=97.0
            (100.5, 102, 100, 101),   # breakout
        ])
        sl, bars_back, source = find_swing_low(o, h, l, c,
                                               breakout_bar_idx=1,
                                               tick_size=0.25,
                                               range_low=99.5,
                                               max_lookback=50, min_sl_ticks=4)
        assert source == "cluster", f"expected cluster, got {source}"
        assert sl == pytest.approx(97.0, abs=0.01)

    def test_no_prior_bars(self):
        o, h, l, c = _bars([(100, 102, 99, 101)])
        sl, _, source = find_swing_low(o, h, l, c,
                                       breakout_bar_idx=0,
                                       tick_size=0.25,
                                       range_low=95.0,
                                       max_lookback=50, min_sl_ticks=0)
        assert source == "no_prior_bars"

    def test_multiple_clusters_uses_most_recent(self):
        """When two clusters exist before breakout, use the most recent."""
        o, h, l, c = _bars([
            (100, 101, 98,  99),   # bar0: down-close cluster A, low=98
            (99,  101, 99, 101),   # bar1: up-close
            (101, 102, 100, 101),  # bar2: up-close
            (101, 102, 99.5, 100), # bar3: down-close cluster B, low=99.5
            (100, 106, 100, 105),  # bar4: breakout
        ])
        sl, bars_back, source = find_swing_low(o, h, l, c,
                                               breakout_bar_idx=4,
                                               tick_size=0.25,
                                               range_low=90.0,
                                               max_lookback=50, min_sl_ticks=0)
        assert source == "cluster"
        assert sl == 99.5, f"expected most-recent cluster low=99.5, got {sl}"


class TestFindSwingHigh:
    def test_basic_up_cluster_for_short(self):
        o, h, l, c = _bars([
            (100, 101, 99, 100),   # bar0: flat
            (100, 103, 100, 102),  # bar1: up-close
            (102, 104, 102, 103),  # bar2: up-close cluster, high=104
            (103, 103, 98,  99),   # bar3: breakout (short)
        ])
        sl, bars_back, source = find_swing_high(o, h, l, c,
                                                breakout_bar_idx=3,
                                                tick_size=0.25,
                                                range_high=105.0,
                                                max_lookback=50, min_sl_ticks=0)
        assert source == "cluster"
        assert sl == 104.0
