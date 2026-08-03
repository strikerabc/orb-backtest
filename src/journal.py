"""
journal.py — assemble one row dict per (EntrySignal, TradeResult, SessionDay).

Every field in the spec's "DATA ACCUMULATION" section is represented here,
plus additional context fields for later filtering.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.entry_detector import EntrySignal
from src.trade_sim import TradeResult
from src.range_builder import SessionDay
from src.regime_sampler import RegimeWindow
from src.config import SESSIONS


def _bar_idx_to_utc(sd: SessionDay, idx: int) -> Optional[str]:
    """Convert absolute bar index to UTC ISO timestamp string."""
    if idx < 0 or idx >= len(sd.bar_timestamps):
        return None
    ns = int(sd.bar_timestamps[idx])
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()


def build_row(
    es: EntrySignal,
    tr: TradeResult,
    sd: SessionDay,
    window: RegimeWindow,
    all_sd_bars_h: np.ndarray,   # full session active-window bars for opp-boundary check
    all_sd_bars_l: np.ndarray,
) -> dict:
    """Return a flat dict representing one trade journal row."""

    tick = sd.tick_size
    sess = SESSIONS[sd.session]
    oh, om = sess["open"]
    open_min = oh * 60 + om
    is_long = es.direction == "long"
    opp_boundary = sd.range_lows[es.range_minutes] if is_long else sd.range_highs[es.range_minutes]

    # Opposite boundary broken flags
    if is_long:
        opp_broke_anywhere   = bool(np.any(all_sd_bars_l <= opp_boundary))
        opp_broke_pre_entry  = bool(np.any(all_sd_bars_l[:es.entry_bar_idx + 1] <= opp_boundary))
    else:
        opp_broke_anywhere   = bool(np.any(all_sd_bars_h >= opp_boundary))
        opp_broke_pre_entry  = bool(np.any(all_sd_bars_h[:es.entry_bar_idx + 1] >= opp_boundary))

    # Time from session open to entry (bar count)
    open_bar = int(np.searchsorted(sd.bar_wall_mins, open_min))
    bars_from_open = es.entry_bar_idx - open_bar

    # Day of week in LOCAL timezone
    dow = pd.Timestamp(sd.local_date).day_name()

    return {
        # ── identifiers ──────────────────────────────────────────────────
        "regime_window":          window.index,
        "regime_start":           str(window.start),
        "regime_end":             str(window.end),
        "date":                   str(sd.local_date),
        "day_of_week":            dow,
        "instrument":             sd.instrument,
        "session":                sd.session,
        # ── range ────────────────────────────────────────────────────────
        "range_minutes":          es.range_minutes,
        "range_high":             sd.range_highs[es.range_minutes],
        "range_low":              sd.range_lows[es.range_minutes],
        "range_width_ticks":      sd.range_widths_ticks[es.range_minutes],
        # ── entry ────────────────────────────────────────────────────────
        "entry_mode":             es.mode,
        "closure_tf":             es.closure_tf,
        "direction":              es.direction,
        "entry_time_utc":         _bar_idx_to_utc(sd, es.entry_bar_idx),
        "entry_price":            es.fill_price,
        "breakout_bar_idx":       es.breakout_bar_idx,
        "tap_in_bar_idx":         es.tap_in_bar_idx,
        "gap_fill":               es.gap_fill,
        "bars_from_open_to_entry": bars_from_open,
        # ── stop loss ────────────────────────────────────────────────────
        "sl_price":               tr.sl_price,
        "sl_distance_ticks":      round(abs(es.fill_price - tr.sl_price) / tick, 2),
        "sl_bars_back":           es.sl_bars_back,
        "sl_source":              es.sl_source,
        # ── take profit / RR ─────────────────────────────────────────────
        "rr":                     tr.rr,
        "tp_price_uncapped":      tr.tp_price_uncapped,
        "tp_price_used":          tr.tp_price_used,
        "r_ticks":                tr.r_ticks,
        "atr_4h":                 sd.atr_4h,
        "tp_to_atr_ratio":        tr.tp_to_atr_ratio,
        "atr_exceeds_cap":        tr.atr_exceeds_cap,
        # ── exit ─────────────────────────────────────────────────────────
        "exit_time_utc":          _bar_idx_to_utc(sd, tr.exit_bar_idx),
        "exit_price":             tr.exit_price,
        "exit_reason":            tr.exit_reason,
        "bars_held":              tr.bars_held,
        # ── results ──────────────────────────────────────────────────────
        "gross_r":                tr.gross_r,
        "net_r":                  tr.net_r,
        "gross_usd":              tr.gross_usd,
        "net_usd":                tr.net_usd,
        "mae_r":                  tr.mae_r,
        "mfe_r":                  tr.mfe_r,
        "same_bar_ambiguous":     tr.same_bar_ambiguous,
        # ── opposite boundary ────────────────────────────────────────────
        "opp_boundary_broken_session":     opp_broke_anywhere,
        "opp_boundary_broken_pre_entry":   opp_broke_pre_entry,
        # ── enrichment ───────────────────────────────────────────────────
        "prev_close":             sd.prev_close,
        "gap_ticks":              sd.gap_ticks,
        "parkinson_vol_14d":      sd.parkinson_vol_14d,
        "realized_vol_14d":       sd.realized_vol_14d,
    }
