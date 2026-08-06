from datetime import date

import numpy as np
import pandas as pd

from src.data_layer import _compute_enrichment
from src.multiplicity import step_down_max_t
from src.regime_sampler import select_windows


def test_daily_volatility_is_lagged():
    dates = pd.date_range("2024-01-01 22:00", periods=20, tz="UTC")
    daily = pd.DataFrame({
        "timestamp": dates,
        "open": np.arange(20) + 100.0,
        "high": np.arange(20) + 102.0,
        "low": np.arange(20) + 99.0,
        "close": np.arange(20) + 101.0,
        "volume": 1,
    })
    minute = pd.DataFrame({
        "timestamp": dates - pd.Timedelta(hours=7),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1,
    })
    before = _compute_enrichment(minute, daily)
    changed = daily.copy()
    changed.loc[15, ["high", "low", "close"]] = [1000.0, 1.0, 900.0]
    after = _compute_enrichment(minute, changed)
    # Day 15's own values cannot change its day-15 enrichment.
    assert before.loc[15, "parkinson_vol_14d"] == after.loc[15, "parkinson_vol_14d"]
    assert before.loc[15, "realized_vol_14d"] == after.loc[15, "realized_vol_14d"]


def test_regime_windows_never_overlap_and_reduce_when_history_is_short():
    windows = select_windows(date(2021, 2, 8), date(2025, 12, 31))
    assert len(windows) < 10
    assert all(a.end < b.start for a, b in zip(windows, windows[1:]))


def test_stepdown_max_t_is_monotone_in_observed_rank():
    observed = np.array([3.0, 2.0, 1.0])
    null = np.array([
        [2.5, 1.0, 0.0],
        [3.5, 2.5, 1.5],
        [0.5, 0.4, 0.3],
    ])
    adjusted = step_down_max_t(observed, null)
    assert np.all(np.diff(adjusted) >= 0)
    assert np.all((adjusted > 0) & (adjusted <= 1))
