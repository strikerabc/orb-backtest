"""
hyp03_width_check.py — verify WIDTH=0pp: ceiling vs structural identity.

Prints band bounds for real vs WIDTH for 10 sample days, then the
contingency table (both/real-only/WIDTH-only/neither triggered) for the
three sub-saturated sessions: BTC/TOK, ETH/TOK, BTC/LDN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import INSTRUMENTS, RANGE_MINUTES, SESSIONS
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.range_builder import build_session_days


def check(sym: str, sess: str, holdout_start: str = "2026-02-01") -> None:
    tick = INSTRUMENTS[sym]["tick_size"]
    df = _compute_enrichment(ensure_data(sym), ensure_daily(sym), tick_size=tick)
    df = df[df["timestamp"] >= pd.Timestamp(holdout_start, tz="UTC")]

    real_days  = build_session_days(df, sym, sess, placebo_mode="real")
    width_days = build_session_days(df, sym, sess, placebo_mode="width")
    byr = {sd.local_date: sd for sd in real_days}
    byw = {sd.local_date: sd for sd in width_days}
    common = sorted(set(byr) & set(byw))

    oh, om   = SESSIONS[sess]["open"]
    open_min = oh * 60 + om
    RM = 5

    # ── band bounds for first 10 days ─────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{sym}/{sess}: real vs WIDTH band (rm={RM}), first 10 days")
    print(f"{'='*90}")
    hdr = f"  {'date':12s}  {'real_rh':>9s}  {'real_rl':>9s}  {'wid_rh':>9s}  {'wid_rl':>9s}"
    hdr += f"  {'center':>9s}  {'midpoint':>9s}  {'Δrh':>7s}  {'Δrl':>7s}  same?"
    print(hdr)
    for d in common[:10]:
        sdr = byr[d]; sdw = byw[d]
        rh = sdr.range_highs.get(RM); rl = sdr.range_lows.get(RM)
        wh = sdw.range_highs.get(RM); wl = sdw.range_lows.get(RM)
        if rh is None or wh is None:
            print(f"  {str(d):12s}  -- rm={RM} missing --")
            continue
        mid = (rh + rl) / 2.0
        w   = sdr.bar_wall_mins
        c   = sdr.bars_c
        rng = (w >= open_min) & (w < open_min + RM)
        center = float(c[rng][-1]) if rng.sum() >= RM else float("nan")
        same = "SAME" if (abs(wh - rh) < tick * 0.01 and
                          abs(wl - rl) < tick * 0.01) else "diff"
        print(f"  {str(d):12s}  {rh:9.2f}  {rl:9.2f}  {wh:9.2f}  {wl:9.2f}"
              f"  {center:9.2f}  {mid:9.2f}  {wh-rh:+7.2f}  {wl-rl:+7.2f}  {same}")

    # ── contingency table ─────────────────────────────────────────────────
    def day_trig(sd: object) -> bool:
        for rm in RANGE_MINUTES:
            rh2 = sd.range_highs.get(rm); rl2 = sd.range_lows.get(rm)
            if rh2 is None:
                continue
            det = open_min + rm
            idx = int(np.searchsorted(sd.bar_wall_mins, det))
            if idx >= len(sd.bars_h):
                continue
            h2 = sd.bars_h[idx:]; l2 = sd.bars_l[idx:]
            if (h2 >= rh2 + tick).any() or (l2 <= rl2 - tick).any():
                return True
        return False

    rt = {d: day_trig(byr[d]) for d in common}
    wt = {d: day_trig(byw[d]) for d in common}
    n       = len(common)
    both    = sum(rt[d] and     wt[d] for d in common)
    r_only  = sum(rt[d] and not wt[d] for d in common)
    w_only  = sum(not rt[d] and wt[d] for d in common)
    neither = sum(not rt[d] and not wt[d] for d in common)

    print(f"\n  Contingency ({n} days):")
    print(f"    both trigger : {both:3d}  ({100*both/n:.1f}%)")
    print(f"    real only    : {r_only:3d}")
    print(f"    WIDTH only   : {w_only:3d}")
    print(f"    neither      : {neither:3d}  ({100*neither/n:.1f}%)")

    if r_only == 0 and w_only == 0:
        print(f"\n  -> CEILING: zero discordant days. Width placebo fires on"
              f" identical days as real.")
        print(f"     Both arms produce {both} trigger days + {neither} non-trigger"
              f" days — perfectly aligned.")
        print(f"     The trigger-rate test had no discriminating power. This is"
              f" explanation (a): saturation / ceiling, not (b): band identity.")
    else:
        print(f"\n  -> DIVERGENCE: {r_only + w_only} discordant days out of {n}.")
        print(f"     Bands are relocating AND producing different outcomes.")


if __name__ == "__main__":
    for sym, sess in [("BTC", "TOK"), ("ETH", "TOK"), ("BTC", "LDN")]:
        check(sym, sess)
