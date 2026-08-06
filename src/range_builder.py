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

from src.config import (
    CONTEXT_BARS_BEFORE_OPEN, RANGE_MINUTES, SESSIONS, INSTRUMENTS,
)

log = logging.getLogger("orb.range")

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class SessionDay:
    """All data needed to run entry detection for one session-day."""
    instrument: str
    session: str
    local_date: date          # local calendar date of session open
    local_tz: str
    # 1-minute bar arrays (numpy, N×4 for OHLC), including context before the
    # session open and continuing through the 11:59 bar.
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
    session_open_idx: int = 0
    context_bars_available: int = 0
    session_bar_completeness: float = 1.0
    contract_changed_in_session: bool = False
    contract_changed_since_prev_session: bool = False
    pct_bars_from_local: float = 0.0
    regime_window: int | None = None


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

    context_start = max(0, open_min - CONTEXT_BARS_BEFORE_OPEN)
    in_sess = (wall >= context_start) & (wall < exit_min)

    df_sess = df_1m[in_sess].copy()
    df_sess["_wall"]    = wall[in_sess]
    df_sess["_locdate"] = loc_date[in_sess]

    days: list[SessionDay] = []
    previous_contract: object | None = None

    for ld, grp in df_sess.groupby("_locdate", sort=True):
        grp = grp.sort_values("timestamp")
        w   = grp["_wall"].to_numpy(copy=True)

        rh_map, rl_map, rw_map = {}, {}, {}
        for rm in RANGE_MINUTES:
            rng_mask = (w >= open_min) & (w < open_min + rm)
            if rng_mask.sum() < rm:   # incomplete range — skip this range size
                continue
            rh = float(grp["high"].to_numpy()[rng_mask].max())
            rl = float(grp["low"].to_numpy()[rng_mask].min())
            rh_map[rm] = rh
            rl_map[rm] = rl
            rw_map[rm] = round((rh - rl) / tick)

        if not rh_map:
            continue   # no valid range sizes today

        # Context plus active window. Trim stale context across a material gap.
        act_mask  = (w >= context_start) & (w <= exit_min - 1)
        act_grp   = grp[act_mask].sort_values("timestamp")

        if len(act_grp) == 0:
            continue

        in_session_grp = act_grp[act_grp["_wall"] >= open_min]
        if in_session_grp.empty:
            continue
        open_pos = int(np.searchsorted(act_grp["_wall"].to_numpy(), open_min))
        if open_pos:
            ts_ns = act_grp["timestamp"].to_numpy("datetime64[ns]").astype("int64")
            gaps = np.flatnonzero(np.diff(ts_ns[:open_pos + 1]) > 5 * 60 * 1_000_000_000)
            if len(gaps):
                act_grp = act_grp.iloc[int(gaps[-1]) + 1:]
                open_pos = int(np.searchsorted(act_grp["_wall"].to_numpy(), open_min))

        expected_bars = max(1, exit_min - open_min)
        completeness = min(1.0, len(in_session_grp) / expected_bars)

        # ATR and enrichment at the session open, never from context bars.
        first_session = in_session_grp.iloc[0]
        atr_val = float(first_session.get("atr_4h", np.nan))

        # Enrichment from first bar's daily-joined columns
        pc  = float(first_session.get("prev_close",        np.nan))
        gap = float(first_session.get("open", np.nan) - pc) / tick if not np.isnan(pc) else np.nan
        pv  = float(first_session.get("parkinson_vol_14d", np.nan))
        rv  = float(first_session.get("realized_vol_14d",  np.nan))

        contracts = in_session_grp.get("_contract", pd.Series(dtype=object)).dropna()
        current_contract = contracts.iloc[0] if len(contracts) else None
        changed_in = bool(contracts.nunique() > 1)
        changed_since = bool(
            previous_contract is not None and current_contract is not None
            and previous_contract != current_contract)
        if len(contracts):
            previous_contract = contracts.iloc[-1]
        sources = in_session_grp.get("_source", pd.Series(dtype=object))
        pct_local = float(sources.eq("local").mean()) if len(sources) else 0.0

        days.append(SessionDay(
            instrument=sym, session=session_name,
            local_date=ld, local_tz=tz,
            bars_o=act_grp["open"].to_numpy(copy=True),
            bars_h=act_grp["high"].to_numpy(copy=True),
            bars_l=act_grp["low"].to_numpy(copy=True),
            bars_c=act_grp["close"].to_numpy(copy=True),
            bars_v=act_grp["volume"].to_numpy(copy=True),
            bar_timestamps=act_grp["timestamp"].to_numpy("datetime64[ns]").astype("int64"),
            bar_wall_mins=act_grp["_wall"].to_numpy(copy=True),
            range_highs=rh_map, range_lows=rl_map,
            range_widths_ticks=rw_map,
            atr_4h=atr_val, tick_size=tick,
            prev_close=pc, gap_ticks=gap,
            parkinson_vol_14d=pv, realized_vol_14d=rv,
            session_open_idx=open_pos,
            context_bars_available=open_pos,
            session_bar_completeness=completeness,
            contract_changed_in_session=changed_in,
            contract_changed_since_prev_session=changed_since,
            pct_bars_from_local=pct_local,
        ))

    log.info("%s %s: %d session-days", sym, session_name, len(days))
    return days
