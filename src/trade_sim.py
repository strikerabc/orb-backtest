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
    ATR_CAP_MULTIPLE, COMMISSION_PER_SIDE_USD, INSTRUMENTS, MIN_TP_TICKS,
    RR_LEVELS, SLIPPAGE_TICKS_BY_SYMBOL, SLIPPAGE_TICKS_BY_SYMBOL_SESSION,
    SLIPPAGE_TICKS_ROUND_TRIP,
)
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
    exit_reason: str           # 'TP' | 'SL' | 'TIME' | 'ATR_INVALIDATED'
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
    comm_ticks = 2.0 * COMMISSION_PER_SIDE_USD / tick_value_usd
    total_ticks = comm_ticks + slip
    return total_ticks / r_ticks


def simulate_trade(
    es: EntrySignal,
    sd: SessionDay,
    rr_levels: list[float] = RR_LEVELS,
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

    if r <= 0 or r_ticks < 1:
        # Degenerate stop — mark all variants invalidated
        return [_invalid_result(rr, entry, sl, r_ticks) for rr in rr_levels]

    # Bars from entry bar onward (inclusive)
    start = es.entry_bar_idx
    h_arr = sd.bars_h[start:]
    l_arr = sd.bars_l[start:]
    o_arr = sd.bars_o[start:]
    c_arr = sd.bars_c[start:]
    nb    = len(h_arr)

    # Precompute running MAE and MFE arrays
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

        exit_price = np.nan
        exit_bar   = nb - 1
        exit_reason = "TIME"
        same_bar_flag = False

        for i in range(nb):
            bar_o = o_arr[i]; bar_h = h_arr[i]; bar_l = l_arr[i]

            # Check gap opens (bar opens beyond SL or TP before price moves)
            if is_long:
                gap_sl = bar_o <= sl     # bar opened at or below stop
                gap_tp = bar_o >= tp_used
            else:
                gap_sl = bar_o >= sl
                gap_tp = bar_o <= tp_used

            if gap_sl:
                exit_price  = bar_o
                exit_reason = "SL"
                exit_bar    = i
                break
            if gap_tp:
                exit_price  = bar_o
                exit_reason = "TP"
                exit_bar    = i
                break

            # Normal intrabar check
            sl_hit = (bar_l <= sl) if is_long else (bar_h >= sl)
            tp_hit = (bar_h >= tp_used) if is_long else (bar_l <= tp_used)

            if sl_hit and tp_hit:
                # Both hit same bar — SL first (conservative)
                exit_price    = sl
                exit_reason   = "SL"
                exit_bar      = i
                same_bar_flag = True
                break
            if sl_hit:
                exit_price  = sl
                exit_reason = "SL"
                exit_bar    = i
                break
            if tp_hit:
                exit_price  = tp_used
                exit_reason = "TP"
                exit_bar    = i
                break

        if np.isnan(exit_price):
            exit_price = c_arr[-1]   # time exit at close of last (11:59) bar

        gross_r   = sign * (exit_price - entry) / r
        net_r     = gross_r - cost_r if gross_r >= 0 else gross_r - cost_r
        gross_usd = gross_r * r_ticks * tv_usd
        net_usd   = net_r   * r_ticks * tv_usd

        mae_r = -float(adverse[:exit_bar + 1].max())  / r   # negative value
        mfe_r =  float(favorable[:exit_bar + 1].max()) / r  # positive value

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
        ))

    return results


def _invalid_result(rr: float, entry: float, sl: float, r_ticks: float) -> TradeResult:
    return TradeResult(
        rr=rr, tp_price_uncapped=np.nan, tp_price_used=np.nan,
        sl_price=sl, r_ticks=r_ticks,
        exit_price=np.nan, exit_bar_idx=-1, exit_reason="INVALID",
        gross_r=np.nan, net_r=np.nan, gross_usd=np.nan, net_usd=np.nan,
        mae_r=np.nan, mfe_r=np.nan, bars_held=0,
        same_bar_ambiguous=False, atr_exceeds_cap=False, tp_to_atr_ratio=None,
        tp_ticks=np.nan, tp_unfillable=False,
    )
