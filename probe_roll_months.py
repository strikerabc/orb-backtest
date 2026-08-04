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
    for sym, ratios in results.items():
        vals = [r for r in ratios if r is not None]
        if not vals:
            print(f"  {sym:<5} no data")
            continue
        gold = sym == "GC"
        d_idx = GOLD_ACTIVE_MONTHS if gold else QUARTERLY_MONTHS
        d_vals = [r for i, r in enumerate(ratios, 1) if r is not None and i in d_idx]
        n_vals = [r for i, r in enumerate(ratios, 1) if r is not None and i not in d_idx]
        dm = np.mean(d_vals) if d_vals else float("nan")
        nm = np.mean(n_vals) if n_vals else float("nan")
        spike = (dm / nm) if (n_vals and nm > 0) else float("nan")
        verdict = ("PERIODIC (delivery-month artefact)"
                   if spike == spike and spike > 1.4 else
                   "UNIFORM deficit" if np.mean(vals) > 1.3 else "fine")
        print(f"  {sym:<5} delivery-month mean {dm:>5.2f} | "
              f"other-month mean {nm:>5.2f} | spike {spike:>5.2f}x  {verdict}")

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
