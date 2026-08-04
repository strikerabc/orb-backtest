"""
diagnose_coverage.py — why do some instruments produce so few trades?

Checks three layers to localise the cause:
  1. DATA      — bars, date range, per-session minute coverage in data/*.parquet
  2. SESSIONS  — session-days surviving the complete-opening-range requirement
  3. TRADES    — per-instrument trade counts, exit_reason mix, variant counts

Key hypothesis: build_session_days() skips a day when the opening range has
fewer than `rm` one-minute bars (range_builder.py: `if rng_mask.sum() < rm`).
Thinly-traded instrument/session pairs have minutes with zero trades and hence
no bar, so the day is dropped entirely. That would show up as high data-bar
counts but very low session-day counts.

Usage:
    python diagnose_coverage.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATA_DIR, INSTRUMENTS, OUTPUTS_DIR, RANGE_MINUTES, SESSIONS

logging.basicConfig(level=logging.WARNING, format="%(message)s")

_DATA = _ROOT / DATA_DIR
_OUT = _ROOT / OUTPUTS_DIR

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def hr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ── 1. DATA COVERAGE ───────────────────────────────────────────────────────
def data_coverage() -> pd.DataFrame:
    hr("1. DATA COVERAGE  (data/<sym>_1m.parquet)")
    rows = []
    for sym, instr in INSTRUMENTS.items():
        p = _DATA / f"{sym}_1m.parquet"
        if not p.exists():
            rows.append({"sym": sym, "status": "MISSING", "bars": 0})
            continue
        df = pd.read_parquet(p, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], utc=True)
        span_days = (ts.max() - ts.min()).days
        rows.append({
            "sym": sym,
            "asset_class": instr.get("asset_class", "?"),
            "status": "ok",
            "bars": len(df),
            "first": ts.min().strftime("%Y-%m-%d"),
            "last": ts.max().strftime("%Y-%m-%d"),
            "span_days": span_days,
            # a 24h-ish future trading ~23h x 5d/7 would be ~7100 bars/week
            "bars_per_cal_day": round(len(df) / max(span_days, 1), 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# ── 2. SESSION-DAY / OPENING-RANGE COMPLETENESS ────────────────────────────
def session_completeness() -> pd.DataFrame:
    hr("2. OPENING-RANGE COMPLETENESS PER INSTRUMENT x SESSION")
    print("For each session, of all local dates that have ANY bar in the")
    print("session window, what fraction have a COMPLETE opening range?")
    print("(complete = >= rm distinct 1-minute bars in the first rm minutes)\n")

    rows = []
    for sym, instr in INSTRUMENTS.items():
        p = _DATA / f"{sym}_1m.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], utc=True)

        for sess_name in instr.get("sessions", list(SESSIONS.keys())):
            sess = SESSIONS[sess_name]
            tz = sess["tz"]
            oh, om = sess["open"]
            open_min = oh * 60 + om

            loc = ts.dt.tz_convert(tz)
            wall = (loc.dt.hour * 60 + loc.dt.minute).to_numpy()
            locdate = loc.dt.date.to_numpy()

            rec = {"sym": sym, "session": sess_name}
            for rm in RANGE_MINUTES:
                m = (wall >= open_min) & (wall < open_min + rm)
                if not m.any():
                    rec[f"or{rm}_days"] = 0
                    rec[f"or{rm}_pct"] = 0.0
                    continue
                s = pd.Series(wall[m]).groupby(pd.Series(locdate[m])).nunique()
                total_days = len(s)
                complete = int((s >= rm).sum())
                rec[f"or{rm}_days"] = complete
                rec[f"or{rm}_pct"] = round(100.0 * complete / max(total_days, 1), 1)
            rows.append(rec)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


# ── 3. TRADE / VARIANT DISTRIBUTION ────────────────────────────────────────
def trade_distribution() -> None:
    hr("3. TRADES AND VARIANTS PER INSTRUMENT  (outputs/)")

    sm_p = _OUT / "summary.parquet"
    tl_p = _OUT / "trade_log.parquet"

    if sm_p.exists():
        sm = pd.read_parquet(sm_p)
        g = (sm.groupby("instrument")
               .agg(variants=("trade_count", "size"),
                    tc_min=("trade_count", "min"),
                    tc_med=("trade_count", "median"),
                    tc_max=("trade_count", "max"),
                    tc_sum=("trade_count", "sum"))
               .reset_index()
               .sort_values("tc_med"))
        print("\nPer-instrument variant trade_count distribution:")
        print(g.to_string(index=False))

        print("\nHow many variants would survive a minimum-sample filter?")
        tot = len(sm)
        for thr in (1, 10, 30, 50, 100, 200):
            n = int((sm["trade_count"] >= thr).sum())
            print(f"  trade_count >= {thr:>4}:  {n:>5} / {tot}  ({100.0*n/tot:5.1f}%)")

    if tl_p.exists():
        tl = pd.read_parquet(tl_p, columns=["instrument", "session", "exit_reason"])
        print("\nExit-reason mix per instrument (% of rows):")
        mix = (pd.crosstab(tl["instrument"], tl["exit_reason"], normalize="index") * 100).round(1)
        print(mix.to_string())

        print("\nRows per instrument x session:")
        print(pd.crosstab(tl["instrument"], tl["session"]).to_string())


def main() -> None:
    data_coverage()
    session_completeness()
    trade_distribution()
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
