"""
trade_sim.py — simulate trade exits for all RR levels given an EntrySignal.

Intrabar ambiguity rule (conservative): if a single 1m bar contains both SL
and TP levels, assume SL is hit first and flag the trade.
Gap fills: if a bar OPENS beyond SL/TP, fill at the bar's open (not the level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import (
    ATR_CAP_MULTIPLE, INSTRUMENTS, MIN_TP_TICKS,
    RR_LEVELS, SLIPPAGE_TICKS_BY_SYMBOL, SLIPPAGE_TICKS_BY_SYMBOL_SESSION,
    SLIPPAGE_TICKS_ROUND_TRIP,
)
from src.contracts import round_trip_commission_usd
from src.entry_detector import EntrySignal
from src.range_builder import SessionDay


@dataclass
class TradeResult:
    rr: float
    tp_price_uncapped: float
    tp_price_used: float       # may equal tp_price_uncapped if under cap
    sl_price: float
    r_ticks: float             # |entry - sl| in ticks
    exit_price: float
    exit_bar_idx: int
    exit_reason: str           # TP | SL | TIME | an explicit invalid reason
    gross_r: float
    net_r: float
    gross_usd: float
    net_usd: float
    mae_r: float               # max adverse excursion in R (≤ 0)
    mfe_r: float               # max favorable excursion in R (≥ 0)
    bars_held: int
    same_bar_ambiguous: bool
    atr_exceeds_cap: bool
    tp_to_atr_ratio: Optional[float]
    tp_ticks: float            # rr * r_ticks — TP distance from entry
    tp_unfillable: bool        # True if tp_ticks < MIN_TP_TICKS (inside spread)
    cost_r: float = np.nan
    gross_r_optimistic: float = np.nan


def slippage_ticks_for(sym: str | None, session: str | None = None) -> float:
    """
    Round-trip slippage in ticks, resolved most-specific-first:

        1. SLIPPAGE_TICKS_BY_SYMBOL_SESSION[(sym, session)]
        2. SLIPPAGE_TICKS_BY_SYMBOL[sym]
        3. SLIPPAGE_TICKS_ROUND_TRIP (global fallback)

    Session specificity matters because spreads differ materially by session,
    and session is a dimension variants are ranked on. One scalar per symbol
    overcharges the liquid session and undercharges the thin one, biasing the
    cross-session comparison itself.
    """
    if sym is None:
        return float(SLIPPAGE_TICKS_ROUND_TRIP)
    if session is not None:
        v = SLIPPAGE_TICKS_BY_SYMBOL_SESSION.get((sym, session))
        if v is not None:
            return float(v)
    return float(SLIPPAGE_TICKS_BY_SYMBOL.get(sym, SLIPPAGE_TICKS_ROUND_TRIP))


def _round_cost_r(r_ticks: float, tick_value_usd: float,
                  sym: str | None = None,
                  session: str | None = None) -> float:
    """
    Total round-trip cost in R units (commission + slippage).

    Slippage is looked up per (symbol, session); one tick is not economically
    comparable across instruments (ETH's tick is 0.0017% of price, ZN's is far
    larger relative to typical stop distance), nor across sessions within one
    instrument.
    """
    if r_ticks <= 0:
        return 0.0
    slip = slippage_ticks_for(sym, session)
    comm_ticks = round_trip_commission_usd(sym or "") / tick_value_usd
    total_ticks = comm_ticks + slip
    return total_ticks / r_ticks


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else len(mask)


def _first_touch_vectorized(
    o: np.ndarray, h: np.ndarray, l: np.ndarray,
    sl: float, tp: float, tick: float, is_long: bool,
) -> tuple[int, float, str, bool]:
    """Return first exit using monotone extrema/searchsorted semantics."""
    n = len(o)
    if is_long:
        run_min_o = np.minimum.accumulate(np.where(np.isnan(o), np.inf, o))
        run_max_o = np.maximum.accumulate(np.where(np.isnan(o), -np.inf, o))
        run_min_l = np.minimum.accumulate(np.where(np.isnan(l), np.inf, l))
        run_max_h = np.maximum.accumulate(np.where(np.isnan(h), -np.inf, h))
        i_gsl = int(np.searchsorted(-run_min_o, -sl, side="left"))
        i_gtp = int(np.searchsorted(run_max_o, tp, side="left"))
        i_sl = int(np.searchsorted(-run_min_l, -sl, side="left"))
        i_tp = int(np.searchsorted(run_max_h, tp + tick, side="left"))
    else:
        run_max_o = np.maximum.accumulate(np.where(np.isnan(o), -np.inf, o))
        run_min_o = np.minimum.accumulate(np.where(np.isnan(o), np.inf, o))
        run_max_h = np.maximum.accumulate(np.where(np.isnan(h), -np.inf, h))
        run_min_l = np.minimum.accumulate(np.where(np.isnan(l), np.inf, l))
        i_gsl = int(np.searchsorted(run_max_o, sl, side="left"))
        i_gtp = int(np.searchsorted(-run_min_o, -tp, side="left"))
        i_sl = int(np.searchsorted(run_max_h, sl, side="left"))
        i_tp = int(np.searchsorted(-run_min_l, -(tp - tick), side="left"))

    i = min(i_gsl, i_gtp, i_sl, i_tp)
    if i >= n:
        return n - 1, np.nan, "TIME", False
    if i_gsl == i:
        return i, float(o[i]), "SL", False
    if i_gtp == i:
        return i, float(o[i]), "TP", False

    sl_hit = bool(l[i] <= sl) if is_long else bool(h[i] >= sl)
    tp_hit = bool(h[i] >= tp + tick) if is_long else bool(l[i] <= tp - tick)
    if sl_hit:
        return i, sl, "SL", tp_hit
    return i, tp, "TP", False


def _first_touch_reference(
    o: np.ndarray, h: np.ndarray, l: np.ndarray,
    sl: float, tp: float, tick: float, is_long: bool,
) -> tuple[int, float, str, bool]:
    """Readable O(bars) oracle retained for randomized equivalence tests."""
    for i, (bar_o, bar_h, bar_l) in enumerate(zip(o, h, l)):
        gap_sl = (bar_o <= sl) if is_long else (bar_o >= sl)
        gap_tp = (bar_o >= tp) if is_long else (bar_o <= tp)
        if gap_sl:
            return i, float(bar_o), "SL", False
        if gap_tp:
            return i, float(bar_o), "TP", False
        sl_hit = (bar_l <= sl) if is_long else (bar_h >= sl)
        tp_hit = (bar_h >= tp + tick) if is_long else (bar_l <= tp - tick)
        if sl_hit:
            return i, sl, "SL", bool(tp_hit)
        if tp_hit:
            return i, tp, "TP", False
    return len(o) - 1, np.nan, "TIME", False


def _simulate_trade(
    es: EntrySignal,
    sd: SessionDay,
    rr_levels: list[float] = RR_LEVELS,
    *,
    reference: bool = False,
) -> list[TradeResult]:
    """
    For each RR level, simulate the exit and return a TradeResult.
    sd.bars_* cover the full active window (from session open to11:59 bar).
    es.entry_bar_idx indexes into sd.bars_*.
    """
    tick     = sd.tick_size
    tv_usd   = INSTRUMENTS[sd.instrument]["tick_value_usd"]
    is_long  = es.direction == "long"
    entry    = es.fill_price
    sl       = es.sl_price
    r        = abs(entry - sl)
    r_ticks  = r / tick
    sign     = 1.0 if is_long else -1.0

    cost_r   = _round_cost_r(r_ticks, tv_usd, sd.instrument, sd.session)
    atr      = sd.atr_4h

    wrong_side = (sl >= entry) if is_long else (sl <= entry)
    if wrong_side or r <= 0 or r_ticks < 1:
        reason = "SL_WRONG_SIDE" if wrong_side else "DEGENERATE_R"
        return [_invalid_result(rr, entry, sl, r_ticks, reason) for rr in rr_levels]

    start = es.entry_bar_idx + (1 if es.fill_at_bar_close else 0)
    assert start >= getattr(sd, "session_open_idx", 0)
    if start >= len(sd.bars_h):
        return [_invalid_result(rr, entry, sl, r_ticks, "NO_HOLD_BARS")
                for rr in rr_levels]
    h_arr = sd.bars_h[start:]
    l_arr = sd.bars_l[start:]
    o_arr = sd.bars_o[start:]
    c_arr = sd.bars_c[start:]
    nb    = len(h_arr)

    if nb == 0:
        return [_invalid_result(rr, entry, sl, r_ticks, "NO_HOLD_BARS")
                for rr in rr_levels]

    if is_long:
        adverse   = entry - l_arr          # positive = adverse
        favorable = h_arr - entry          # positive = favorable
    else:
        adverse   = h_arr - entry
        favorable = entry - l_arr

    results: list[TradeResult] = []

    for rr in rr_levels:
        tp_raw = entry + sign * rr * r
        # TP distance in ticks. Below MIN_TP_TICKS the target sits inside the
        # spread and cannot fill, yet the exit walk would record it as a win.
        tp_ticks_val = rr * r_ticks
        tp_unfillable_flag = bool(tp_ticks_val < MIN_TP_TICKS)
        tp_to_atr = (abs(tp_raw - entry) / atr) if (atr and not np.isnan(atr) and atr > 0) else None
        exceeds_cap = bool(tp_to_atr is not None and tp_to_atr > ATR_CAP_MULTIPLE)
        tp_used = tp_raw   # we always simulate; mark flag for post-analysis filtering

        finder = _first_touch_reference if reference else _first_touch_vectorized
        exit_bar, exit_price, exit_reason, same_bar_flag = finder(
            o_arr, h_arr, l_arr, sl, tp_used, tick, is_long)

        if np.isnan(exit_price):
            exit_price = c_arr[-1]   # time exit at close of last (11:59) bar

        gross_r   = sign * (exit_price - entry) / r
        net_r     = gross_r - cost_r
        gross_usd = gross_r * r_ticks * tv_usd
        net_usd   = net_r   * r_ticks * tv_usd

        mae_r = -max(0.0, float(adverse[:exit_bar + 1].max())) / r
        mfe_r = max(0.0, float(favorable[:exit_bar + 1].max())) / r
        gross_r_optimistic = rr if same_bar_flag else gross_r

        results.append(TradeResult(
            rr=rr, tp_price_uncapped=tp_raw, tp_price_used=tp_used,
            sl_price=sl, r_ticks=r_ticks,
            exit_price=exit_price, exit_bar_idx=start + exit_bar,
            exit_reason=exit_reason,
            gross_r=round(gross_r, 4), net_r=round(net_r, 4),
            gross_usd=round(gross_usd, 2), net_usd=round(net_usd, 2),
            mae_r=round(mae_r, 4), mfe_r=round(mfe_r, 4),
            bars_held=exit_bar + 1,
            same_bar_ambiguous=same_bar_flag,
            atr_exceeds_cap=exceeds_cap,
            tp_to_atr_ratio=round(tp_to_atr, 4) if tp_to_atr else None,
            tp_ticks=round(tp_ticks_val, 4),
            tp_unfillable=tp_unfillable_flag,
            cost_r=round(cost_r, 4),
            gross_r_optimistic=round(gross_r_optimistic, 4),
        ))

    return results


def simulate_trade(
    es: EntrySignal,
    sd: SessionDay,
    rr_levels: list[float] = RR_LEVELS,
) -> list[TradeResult]:
    return _simulate_trade(es, sd, rr_levels, reference=False)


def _simulate_trade_reference(
    es: EntrySignal,
    sd: SessionDay,
    rr_levels: list[float] = RR_LEVELS,
) -> list[TradeResult]:
    return _simulate_trade(es, sd, rr_levels, reference=True)


def _invalid_result(rr: float, entry: float, sl: float, r_ticks: float,
                    reason: str = "INVALID") -> TradeResult:
    return TradeResult(
        rr=rr, tp_price_uncapped=np.nan, tp_price_used=np.nan,
        sl_price=sl, r_ticks=r_ticks,
        exit_price=np.nan, exit_bar_idx=-1, exit_reason=reason,
        gross_r=np.nan, net_r=np.nan, gross_usd=np.nan, net_usd=np.nan,
        mae_r=np.nan, mfe_r=np.nan, bars_held=0,
        same_bar_ambiguous=False, atr_exceeds_cap=False, tp_to_atr_ratio=None,
        tp_ticks=np.nan, tp_unfillable=False,
        cost_r=np.nan, gross_r_optimistic=np.nan,
    )
