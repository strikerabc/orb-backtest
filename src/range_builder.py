"""
range_builder.py — extract session-day windows and opening-range boundaries.

All session logic anchors to LOCAL timezone (DST-correct via pytz/zoneinfo).
The exit bar is the 11:59 bar in local time (close = 12:00:00 local exactly).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RANGE_MINUTES, SESSIONS, INSTRUMENTS

log = logging.getLogger("orb.range")

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class SessionDay:
    """All data needed to run entry detection for one session-day."""
    instrument: str
    session: str
    local_date: date          # local calendar date of session open
    local_tz: str
    # 1-minute bar arrays (numpy, N×4 for OHLC) over active window
    # active window = range_end through 11:59 bar inclusive
    bars_o: np.ndarray
    bars_h: np.ndarray
    bars_l: np.ndarray
    bars_c: np.ndarray
    bars_v: np.ndarray
    bar_timestamps: np.ndarray   # UTC int64 ns for reference
    bar_wall_mins: np.ndarray    # minutes from midnight in LOCAL tz per bar
    # range boundaries by size
    range_highs: dict[int, float]   # {5: rh, 15: rh, 30: rh}
    range_lows:  dict[int, float]
    range_widths_ticks: dict[int, float]
    # context
    atr_4h: float
    tick_size: float
    # enrichment (may be NaN)
    prev_close: float
    gap_ticks: float
    parkinson_vol_14d: float
    realized_vol_14d: float


def _wall_mins(ts_series: pd.Series, tz: str) -> np.ndarray:
    """Return minutes-from-midnight in LOCAL tz for each UTC timestamp."""
    loc = ts_series.dt.tz_convert(tz)
    return (loc.dt.hour * 60 + loc.dt.minute).to_numpy(copy=True)


def build_session_days(
    df_1m: pd.DataFrame,
    sym: str,
    session_name: str,
) -> list[SessionDay]:
    """
    Return a SessionDay for every trading day where the session has data.
    Only days with a complete opening range (first N minutes all present)
    for at least one range size are included.
    """
    sess     = SESSIONS[session_name]
    tz       = sess["tz"]
    oh, om   = sess["open"]
    open_min = oh * 60 + om          # e.g. 570 for 09:30
    exit_min = sess["exit"][0] * 60 + sess["exit"][1]   # 720 for 12:00
    tick     = INSTRUMENTS[sym]["tick_size"]

    # ── local wall-clock minutes for every bar ─────────────────────────────
    wall = _wall_mins(df_1m["timestamp"], tz)

    # ── group bars into local calendar days (the session's local date) ──────
    loc_ts   = df_1m["timestamp"].dt.tz_convert(tz)
    loc_date = loc_ts.dt.date.to_numpy(copy=True)

    # Mask: bars relevant to this session (open_min ≤ wall < exit_min + 1)
    in_sess = (wall >= open_min) & (wall < exit_min)

    df_sess = df_1m[in_sess].copy()
    df_sess["_wall"]    = wall[in_sess]
    df_sess["_locdate"] = loc_date[in_sess]

    days: list[SessionDay] = []

    for ld, grp in df_sess.groupby("_locdate", sort=True):
        grp = grp.sort_values("timestamp")
        w   = grp["_wall"].to_numpy(copy=True)

        rh_map, rl_map, rw_map = {}, {}, {}
        for rm in RANGE_MINUTES:
            rng_mask = (w >= open_min) & (w < open_min + rm)
            if rng_mask.sum() < rm:   # incomplete range — skip this range size
                continue
            rh = float(grp.loc[grp.index[rng_mask], "high"].max())
            rl = float(grp.loc[grp.index[rng_mask], "low"].min())
            rh_map[rm] = rh
            rl_map[rm] = rl
            rw_map[rm] = round((rh - rl) / tick)

        if not rh_map:
            continue   # no valid range sizes today

        # Active window: from open_min to exit_min - 1 (11:59) inclusive
        act_mask  = (w >= open_min) & (w <= exit_min - 1)
        act_grp   = grp[act_mask].sort_values("timestamp")

        if len(act_grp) == 0:
            continue

        # ATR at session open (first bar's precomputed value)
        atr_val = float(grp.iloc[0].get("atr_4h", np.nan))

        # Enrichment from first bar's daily-joined columns
        pc  = float(grp.iloc[0].get("prev_close",        np.nan))
        gap = float(grp.iloc[0].get("open", np.nan) - pc) / tick if not np.isnan(pc) else np.nan
        pv  = float(grp.iloc[0].get("parkinson_vol_14d", np.nan))
        rv  = float(grp.iloc[0].get("realized_vol_14d",  np.nan))

        days.append(SessionDay(
            instrument=sym, session=session_name,
            local_date=ld, local_tz=tz,
            bars_o=act_grp["open"].to_numpy(copy=True),
            bars_h=act_grp["high"].to_numpy(copy=True),
            bars_l=act_grp["low"].to_numpy(copy=True),
            bars_c=act_grp["close"].to_numpy(copy=True),
            bars_v=act_grp["volume"].to_numpy(copy=True),
            bar_timestamps=act_grp["timestamp"].view("int64").to_numpy(copy=True)
                           if hasattr(act_grp["timestamp"], 'view')
                           else act_grp["timestamp"].astype("int64").to_numpy(copy=True),
            bar_wall_mins=act_grp["_wall"].to_numpy(copy=True),
            range_highs=rh_map, range_lows=rl_map,
            range_widths_ticks=rw_map,
            atr_4h=atr_val, tick_size=tick,
            prev_close=pc, gap_ticks=gap,
            parkinson_vol_14d=pv, realized_vol_14d=rv,
        ))

    log.info("%s %s: %d session-days", sym, session_name, len(days))
    return days
