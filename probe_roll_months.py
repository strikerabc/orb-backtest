"""
probe_roll_months.py — is .c.0 degradation periodic (delivery months)?

Why this exists
---------------
probe_all_rolls.py sampled ONE month (2024-06) and had two defects:

  1. It swallowed API errors with `except Exception: None`, so three cells
     printed "err" with no reason. CL.c.0 was among them -- yet CL has
     2,507,503 cached bars downloaded with that exact symbol, proving "err"
     does not mean "symbol invalid". Almost certainly rate limiting from 30
     rapid metadata calls. This script captures error text and retries with
     backoff.

  2. June 2024 is a DELIVERY month for the H/M/U/Z cycle (Treasuries, FX) and
     for gold's G/J/M/Q/V/Z cycle. Treasuries/FX/metals roll BEFORE the
     delivery month opens, so .c.0 spent all of June pointing at a contract
     in delivery that barely trades. Equity index futures roll only ~1 week
     pre-expiry, which is why ES measured 1.03x while ZN measured 2.2x.

     A single delivery-month sample therefore shows WORST CASE for quarterly
     contracts and says nothing about the annual average.

What this measures
------------------
cost(.c.0) vs cost(.v.0) across 12 months of 2024. If the ratio spikes on a
quarterly rhythm, the delivery-month mechanism is confirmed and the annual
mean gives the true data deficit.

Note on the decision: even a modest annual average is disqualifying if the
deficit is CONCENTRATED in specific calendar months, because that biases the
backtest by period rather than adding uniform noise.

Metadata calls only. Downloads nothing, spends no credits.

Usage:
    $env:DATABENTO_API_KEY = "..."
    python probe_roll_months.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATASET, INSTRUMENTS, SCHEMA_1M, STYPE

# Suspects + controls. ES is the control that measured 1.03x in June.
SYMS = ["GC", "ZN", "6J", "CL", "6E", "ES"]
SUFFIXES = ["c", "v"]

MONTHS = [(f"2024-{m:02d}-01",
           f"2024-{m+1:02d}-01" if m < 12 else "2025-01-01")
          for m in range(1, 13)]

# CME delivery-month cycles
QUARTERLY = {"H", "M", "U", "Z"}          # Mar Jun Sep Dec -> ZN, 6J, ES
GOLD_MONTHS = {"G", "J", "M", "Q", "V", "Z"}  # Feb Apr Jun Aug Oct Dec
QUARTERLY_MONTHS = {3, 6, 9, 12}
GOLD_ACTIVE_MONTHS = {2, 4, 6, 8, 10, 12}

MAX_RETRIES = 4
BASE_SLEEP = 0.6


def hr(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def get_cost(client, symbol: str, start: str, end: str) -> tuple[float | None, str]:
    """Cost with retry/backoff. Returns (cost, error_text)."""
    last = ""
    for attempt in range(MAX_RETRIES):
        try:
            c = client.metadata.get_cost(
                dataset=DATASET, symbols=[symbol], schema=SCHEMA_1M,
                start=start, end=end, stype_in=STYPE,
            )
            return float(c), ""
        except Exception as exc:
            last = str(exc).split("\n")[0][:90]
            time.sleep(BASE_SLEEP * (2 ** attempt))
    return None, last


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set")
        sys.exit(1)
    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(key=api_key)

    hr("MONTHLY .c.0 / .v.0 RATIO ACROSS 2024")
    print("ratio = cost(.v.0) / cost(.c.0).  1.0 = .c.0 is fine.")
    print("D marks a delivery month for that product's cycle.\n")

    header = "  sym   " + "".join(f"{m:>7}" for m in range(1, 13)) + "     mean    max"
    print(header)
    print("  " + "-" * (len(header) - 2))

    errors: list[str] = []
    results: dict[str, list[float | None]] = {}

    for sym in SYMS:
        if sym not in INSTRUMENTS:
            continue
        root = INSTRUMENTS[sym]["continuous_symbol"].split(".")[0]
        ratios: list[float | None] = []

        for mi, (start, end) in enumerate(MONTHS, start=1):
            cc, e1 = get_cost(client, f"{root}.c.0", start, end)
            cv, e2 = get_cost(client, f"{root}.v.0", start, end)
            if e1:
                errors.append(f"{root}.c.0 {start}: {e1}")
            if e2:
                errors.append(f"{root}.v.0 {start}: {e2}")
            ratios.append((cv / cc) if (cc and cv and cc > 0) else None)

        results[sym] = ratios
        vals = [r for r in ratios if r is not None]

        cells = ""
        for mi, r in enumerate(ratios, start=1):
            is_d = (mi in GOLD_ACTIVE_MONTHS) if sym == "GC" else (mi in QUARTERLY_MONTHS)
            mark = "D" if is_d else " "
            cells += f"{('--' if r is None else f'{r:.1f}'):>6}{mark}"

        mean_s = f"{np.mean(vals):.2f}" if vals else "--"
        max_s = f"{np.max(vals):.1f}" if vals else "--"
        print(f"  {sym:<5} {cells}  {mean_s:>7} {max_s:>6}")

    hr("INTERPRETATION")
    print("Verdict order matters. Absolute severity is checked BEFORE")
    print("periodicity, and the worst months are found EMPIRICALLY rather than")
    print("assumed to be delivery months.")
    print()
    print("Both corrections come from this script's first run mislabelling:")
    print("  GC     -> called PERIODIC because spike 2.19x cleared a 1.4")
    print("            threshold, when in fact min ratio was 16.3x: broken in")
    print("            EVERY month. Severity must be tested first.")
    print("  6E/6J  -> called UNIFORM because spike came out at 0.22x (below")
    print("            threshold) since their delivery months are the BEST.")
    print("            They are periodic with inverted phase, so periodicity")
    print("            must be tested in both directions.")
    print()

    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for sym, ratios in results.items():
        vals = [r for r in ratios if r is not None]
        if not vals:
            print(f"  {sym:<5} no data")
            continue

        lo, hi, mean = float(np.min(vals)), float(np.max(vals)), float(np.mean(vals))
        n_bad = sum(1 for r in vals if r >= 1.4)
        # empirical worst months, not assumed
        ranked = sorted(((r, i) for i, r in enumerate(ratios, 1) if r is not None),
                        reverse=True)
        worst = ", ".join(MONTH_NAMES[i - 1] for _, i in ranked[:4])

        # 1. absolute severity first -- a high floor means always broken
        if lo >= 3.0:
            verdict = f"BROKEN IN EVERY MONTH (min {lo:.1f}x)"
        # 2. periodicity in EITHER direction
        elif hi / max(lo, 1e-9) >= 1.8 and hi >= 1.4:
            verdict = f"PERIODIC ({n_bad}/{len(vals)} months bad, worst: {worst})"
        # 3. uniform elevation
        elif mean > 1.3:
            verdict = f"UNIFORM deficit ({mean:.2f}x)"
        else:
            verdict = "fine"

        print(f"  {sym:<5} min {lo:>6.1f}x | mean {mean:>6.2f}x | max {hi:>6.1f}x | "
              f"bad months {n_bad:>2}/{len(vals):<2}  {verdict}")

    hr("ACTION")
    need = []
    for sym, ratios in results.items():
        vals = [r for r in ratios if r is not None]
        if vals and (float(np.mean(vals)) > 1.3 or float(np.max(vals)) >= 1.8):
            need.append((sym, float(np.mean(vals)), float(np.max(vals))))
    if need:
        print("  Switch continuous_symbol to .v.0 and re-download:")
        for sym, mean, hi in sorted(need, key=lambda t: -t[1]):
            print(f"    {sym:<5} mean {mean:>6.2f}x  max {hi:>6.1f}x")
        print("\n  Cache paths are keyed by roll type, so changing the symbol in")
        print("  config.py routes to a new file and triggers a fresh download.")
        print("  Old .c.0 files are left on disk untouched.")
    else:
        print("  No instrument requires re-download.")

    if errors:
        hr(f"ERRORS AFTER {MAX_RETRIES} RETRIES ({len(errors)})")
        seen = set()
        for e in errors:
            k = e.split(":")[-1].strip()[:60]
            if k not in seen:
                seen.add(k)
                print(f"  {e}")
        print("\n  Persistent errors here are real; transient ones were retried away.")
    else:
        hr("NO ERRORS — every cell resolved after retries")
        print("  Confirms the earlier 'err' cells were rate limiting, not")
        print("  invalid symbols. CL and 6E are now measured.")

    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
