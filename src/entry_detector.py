"""
entry_detector.py — vectorized II / CC / TI / R-II / R-CC detection.

Entry modes operate on one SessionDay at a time. All detection is done with
numpy argmax (no per-bar Python loops). CC/R-CC resample 1m bars to the
requested closure timeframe aligned to session open.

Returns a list of EntrySignal; at most one per (range_size, mode, closure_tf,
direction) — first trigger in the session (A12).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import CLOSURE_TFS, RANGE_MINUTES, SESSIONS
from src.range_builder import SessionDay
from src.swing_detector import find_swing_low, find_swing_high


@dataclass
class EntrySignal:
    mode: str            # 'II', 'CC', 'TI', 'R-II', 'R-CC'
    closure_tf: int      # minutes; 1 for II/TI/R-II
    range_minutes: int
    direction: str       # 'long' or 'short'
    entry_bar_idx: int   # index into sd.bars_* active-window arrays
    fill_price: float
    breakout_bar_idx: int
    tap_in_bar_idx: Optional[int]
    boundary: float      # range_high (long) or range_low (short)
    sl_price: float
    sl_bars_back: int
    sl_source: str
    gap_fill: bool       # True if bar opened past trigger level
    fill_at_bar_close: bool = False


def _resample_to_tf(open_min_wall: int, bars_c: np.ndarray,
                    bar_wall_mins: np.ndarray, tf: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample 1m close array to `tf`-minute bars aligned to session open.
    Returns (tf_closes, tf_bar_last_1m_idx) where tf_bar_last_1m_idx[i]
    is the index of the last 1m bar belonging to each tf-bar.
    """
    # Bucket index for each 1m bar
    elapsed = bar_wall_mins - open_min_wall   # minutes since session open
    bucket  = elapsed // tf                   # which tf-bar each 1m bar belongs to

    n_buckets = int(bucket.max()) + 1 if len(bucket) > 0 else 0
    tf_last   = np.full(n_buckets, -1, dtype=np.int32)
    if n_buckets:
        np.maximum.at(tf_last, bucket.astype(np.intp), np.arange(len(bucket)))
    tf_closes = np.full(n_buckets, np.nan)
    present = tf_last >= 0
    tf_closes[present] = bars_c[tf_last[present]]
    expected_last = open_min_wall + np.arange(n_buckets) * tf + tf - 1
    complete = present & (bar_wall_mins[np.maximum(tf_last, 0)] == expected_last)
    tf_closes[~complete] = np.nan
    tf_last[~complete] = -1

    return tf_closes, tf_last


def detect_entries(sd: SessionDay) -> list[EntrySignal]:
    """Detect all triggered entry signals for a single session-day."""
    signals: list[EntrySignal] = []
    tick  = sd.tick_size
    sess  = SESSIONS[sd.session]
    oh, om = sess["open"]
    open_min = oh * 60 + om

    for rm in RANGE_MINUTES:
        if rm not in sd.range_highs:
            continue

        rh = sd.range_highs[rm]
        rl = sd.range_lows[rm]

        # Range-end index: first active bar at or after open+rm
        range_end_wall = open_min + rm
        active_start   = int(np.searchsorted(sd.bar_wall_mins, range_end_wall))
        if active_start >= len(sd.bars_h):
            continue

        h    = sd.bars_h[active_start:]
        l    = sd.bars_l[active_start:]
        c    = sd.bars_c[active_start:]
        o    = sd.bars_o[active_start:]
        wall = sd.bar_wall_mins[active_start:]
        N    = len(h)
        if N == 0:
            continue

        # Include range bars for swing lookback (before breakout)
        all_o = sd.bars_o; all_h = sd.bars_h; all_l = sd.bars_l; all_c = sd.bars_c

        for direction in ("long", "short"):
            boundary = rh if direction == "long" else rl
            is_long  = direction == "long"

            # ── II: first bar where H ≥ rh+tick (long) or L ≤ rl-tick (short)
            ii_trigger = (h >= rh + tick) if is_long else (l <= rl - tick)
            if ii_trigger.any():
                bi   = int(np.argmax(ii_trigger))
                babs = active_start + bi            # absolute bar index
                fill = float(max(rh + tick, o[bi])) if is_long else float(min(rl - tick, o[bi]))
                sl, slb, sls = (find_swing_low(all_o, all_h, all_l, all_c, babs,
                                               tick, rl, entry_price=fill) if is_long else
                                find_swing_high(all_o, all_h, all_l, all_c, babs,
                                                tick, rh, entry_price=fill))
                signals.append(EntrySignal(
                    mode="II", closure_tf=1, range_minutes=rm,
                    direction=direction, entry_bar_idx=babs,
                    fill_price=fill, breakout_bar_idx=babs,
                    tap_in_bar_idx=None, boundary=boundary,
                    sl_price=sl, sl_bars_back=slb, sl_source=sls,
                    gap_fill=(o[bi] > rh + tick if is_long else o[bi] < rl - tick),
                ))

            # Breakout bar index for TI/R-II/R-CC (same as II)
            brk_rel  = int(np.argmax(ii_trigger)) if ii_trigger.any() else -1
            brk_abs  = (active_start + brk_rel) if brk_rel >= 0 else -1

            # ── CC and R-CC ────────────────────────────────────────────────
            for tf in CLOSURE_TFS[rm]:
                if tf == 1:
                    cc_c = c; cc_last = np.arange(N, dtype=np.int32)
                else:
                    cc_c, cc_last = _resample_to_tf(open_min + rm, c, wall, tf)

                cc_trigger = (cc_c > rh) if is_long else (cc_c < rl)

                if cc_trigger.any():
                    ti_cc = int(np.argmax(cc_trigger))
                    last1m = int(cc_last[ti_cc])
                    abs_entry = active_start + last1m
                    fill_cc  = float(cc_c[ti_cc])
                    sl, slb, sls = (find_swing_low(all_o, all_h, all_l, all_c,
                                                   max(brk_abs, 0) if brk_abs >= 0 else abs_entry,
                                                   tick, rl, entry_price=fill_cc) if is_long else
                                    find_swing_high(all_o, all_h, all_l, all_c,
                                                    max(brk_abs, 0) if brk_abs >= 0 else abs_entry,
                                                    tick, rh, entry_price=fill_cc))
                    signals.append(EntrySignal(
                        mode="CC", closure_tf=tf, range_minutes=rm,
                        direction=direction, entry_bar_idx=abs_entry,
                        fill_price=fill_cc, breakout_bar_idx=brk_abs,
                        tap_in_bar_idx=None, boundary=boundary,
                        sl_price=sl, sl_bars_back=slb, sl_source=sls,
                        gap_fill=False, fill_at_bar_close=True,
                    ))

            if brk_abs < 0:
                continue  # no breakout → TI/R-II/R-CC not possible

            # Post-breakout bars for tap-in search
            post = brk_rel + 1
            if post >= N:
                continue

            # ── TI: first bar after breakout where price returns to boundary
            ti_cond = (l[post:] <= rh) if is_long else (h[post:] >= rl)
            if not ti_cond.any():
                continue
            ti_rel  = post + int(np.argmax(ti_cond))
            ti_abs  = active_start + ti_rel
            ti_fill = boundary

            sl_ti, slb_ti, sls_ti = (
                find_swing_low(all_o, all_h, all_l, all_c, brk_abs, tick, rl,
                               entry_price=ti_fill)
                if is_long else
                find_swing_high(all_o, all_h, all_l, all_c, brk_abs, tick, rh,
                                entry_price=ti_fill)
            )
            signals.append(EntrySignal(
                mode="TI", closure_tf=1, range_minutes=rm,
                direction=direction, entry_bar_idx=ti_abs,
                fill_price=ti_fill, breakout_bar_idx=brk_abs,
                tap_in_bar_idx=ti_abs, boundary=boundary,
                sl_price=sl_ti, sl_bars_back=slb_ti, sl_source=sls_ti,
                gap_fill=False,
            ))

            # Post-tap-in bars for R-II / R-CC
            after_ti = ti_rel + 1
            if after_ti >= N:
                continue

            # ── R-II
            rii_cond = (h[after_ti:] >= rh + tick) if is_long else (l[after_ti:] <= rl - tick)
            if rii_cond.any():
                rii_rel = after_ti + int(np.argmax(rii_cond))
                rii_abs = active_start + rii_rel
                rii_fill = (float(max(rh + tick, o[rii_rel])) if is_long
                            else float(min(rl - tick, o[rii_rel])))
                sl_r, slb_r, sls_r = (
                    find_swing_low(all_o, all_h, all_l, all_c, brk_abs, tick, rl,
                                   entry_price=rii_fill)
                    if is_long else
                    find_swing_high(all_o, all_h, all_l, all_c, brk_abs, tick, rh,
                                    entry_price=rii_fill)
                )
                signals.append(EntrySignal(
                    mode="R-II", closure_tf=1, range_minutes=rm,
                    direction=direction, entry_bar_idx=rii_abs,
                    fill_price=rii_fill, breakout_bar_idx=brk_abs,
                    tap_in_bar_idx=ti_abs, boundary=boundary,
                    sl_price=sl_r, sl_bars_back=slb_r, sl_source=sls_r,
                    gap_fill=(o[rii_rel] > rh + tick if is_long else o[rii_rel] < rl - tick),
                ))

            # ── R-CC
            for tf in CLOSURE_TFS[rm]:
                c_ati = c[after_ti:]; wall_ati = wall[after_ti:]
                if tf == 1:
                    rcc_c = c_ati; rcc_last = np.arange(len(c_ati), dtype=np.int32)
                else:
                    rcc_c, rcc_last = _resample_to_tf(open_min + rm, c_ati, wall_ati, tf)

                rcc_trigger = (rcc_c > rh) if is_long else (rcc_c < rl)
                if rcc_trigger.any():
                    rcc_ti   = int(np.argmax(rcc_trigger))
                    last1m_r = int(rcc_last[rcc_ti])
                    rcc_abs  = active_start + after_ti + last1m_r
                    rcc_fill = float(rcc_c[rcc_ti])
                    sl_rc, slb_rc, sls_rc = (
                        find_swing_low(all_o, all_h, all_l, all_c, brk_abs, tick, rl,
                                       entry_price=rcc_fill)
                        if is_long else
                        find_swing_high(all_o, all_h, all_l, all_c, brk_abs, tick, rh,
                                        entry_price=rcc_fill)
                    )
                    signals.append(EntrySignal(
                        mode="R-CC", closure_tf=tf, range_minutes=rm,
                        direction=direction, entry_bar_idx=rcc_abs,
                        fill_price=rcc_fill,
                        breakout_bar_idx=brk_abs,
                        tap_in_bar_idx=ti_abs, boundary=boundary,
                        sl_price=sl_rc, sl_bars_back=slb_rc, sl_source=sls_rc,
                        gap_fill=False, fill_at_bar_close=True,
                    ))

    return signals
